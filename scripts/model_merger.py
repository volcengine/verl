# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict
import re
import os
import torch
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq
from concurrent.futures import ThreadPoolExecutor
from torch.distributed._tensor import DTensor, Shard, Placement


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', required=True, type = str, help="The path for your saved model")
    parser.add_argument("--hf_upload_path", default=False, type = str, help="The path of the huggingface repo to upload")
    args = parser.parse_args()

    assert not args.local_dir.endswith("huggingface"), "The local_dir should not end with huggingface"
    local_dir = args.local_dir

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)  
            break  
    assert world_size, "No model file with the proper format"
        
    state_dict = torch.load(os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt'), map_location='cpu')
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]
    assert isinstance(weight, torch.distributed._tensor.DTensor)
    # get sharding info
    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    print(f'Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}')

    assert mesh_dim_names in (
        ('fsdp',),
    ), f'Unsupported mesh_dim_names {mesh_dim_names}'

    if 'tp' in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f'Processing model shards with {total_shards} {mesh_shape} in total')

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank):
        model_path = os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank)
    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except:
                print("-"*30)
                print(model_state_dict)
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == 'dp':
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key] = tensor.bfloat16()

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        # merge shards
        placements: Tuple[Shard] = param_placements[key]
        if len(mesh_shape) == 1:
            # 1-D list, FSDP without TP
            assert len(placements) == 1
            shards = state_dict[key]
            state_dict[key] = merge_by_placement(shards, placements[0])
        else:
            # 2-D list, FSDP + TP
            raise NotImplementedError("FSDP + TP is not supported yet")

    print('Writing to local disk')
    hf_path = os.path.join(local_dir, 'huggingface')
    config = AutoConfig.from_pretrained(hf_path)

    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif 'ForConditionalGeneration' in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f'Unknown architecture {config["architectures"]}')

    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Saving model to {hf_path}')
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict
    del model
    if args.hf_upload_path:
        # Push to hugging face
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.hf_upload_path, private=False, exist_ok=True)
        api.upload_folder(
            folder_path=hf_path,
            repo_id=args.hf_upload_path,
            repo_type="model"
        )
    
    





