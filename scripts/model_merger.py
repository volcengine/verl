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
from safetensors.torch import load_file

from verl.utils.megatron_utils import get_model_checkpoint_path, get_hf_model_checkpoint_path

parser = argparse.ArgumentParser()
parser.add_argument('--backend', type = str, required=True, help="The backend of the model")
parser.add_argument('--tie-word-embedding', action='store_true', help="Whether to tie word embedding weights")
parser.add_argument('--is-value-model', action='store_true', help="Whether the model loaded as value model")
parser.add_argument('--hf_model_path', type = str, required=True, help="The path for the huggingface model")
parser.add_argument('--local_dir', type = str, required=True, help="The path for your saved model. For megatron, point to the base dir of model, rng, optimizer checkpoints, commonly be `config.default_local_dir/global_step_\{global_step\}`.")
parser.add_argument('--target_dir', required=False, default="tmp", type = str, help="The path for the target model")
parser.add_argument("--hf_upload_path", default=False, type = str, help="The path of the huggingface repo to upload")
parser.add_argument("--test", action="store_true", help="test correctness of hf_model")
parser.add_argument("--test_hf_dir", type = str, required=False, help="test correctness of hf_model, , with hf_model in checkpoint.contents")
args = parser.parse_args()
os.makedirs(args.target_dir, exist_ok=True)
if args.test:
    assert args.test_hf_dir is not None, f'You must run verl save checkpoint first, with hf_model in checkpoint.contents, and provide the directory here'

def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def upload_model_to_huggingface(hf_path):
    # Push to hugging face
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=args.hf_upload_path, private=False, exist_ok=True)
    api.upload_folder(
        folder_path=hf_path,
        repo_id=args.hf_upload_path,
        repo_type="model"
    )
    

def convert_fsdp_checkpoints_to_hfmodels():
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
    if args.target_dir is None:
        hf_path = os.path.join(local_dir, 'huggingface')
    else:
        hf_path = args.target_dir
    config = AutoConfig.from_pretrained(args.hf_model_path)

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
        upload_model_to_huggingface(hf_path)


def get_tp_pp_rank_from_sharded_dir(sharded_dir):
    match = re.match(r"mp_rank_(\d\d)_(\d\d\d)", sharded_dir)
    tp_rank = int(match.group(1))
    pp_rank = int(match.group(2))
    return tp_rank, pp_rank

def check_megatron_checkpoint_path(model_path):
    sharded_dirs = sorted(os.listdir(model_path))
    tp_size = 0
    pp_size = 0
    for sharded_dir in sharded_dirs:
        match = re.match(r"mp_rank_(\d\d)_(\d\d\d)", sharded_dir)
        assert match, f"Invalid sharded dir {sharded_dir}"
        assert f"model.pt" in os.listdir(os.path.join(model_path, sharded_dir)), f"model.pt not found in {sharded_dir}"
        tp_rank = int(match.group(1))
        pp_rank = int(match.group(2))
        if tp_size < tp_rank + 1:
            tp_size = tp_rank + 1
        if pp_size < pp_rank + 1:
            pp_size = pp_rank + 1
    return sharded_dirs, tp_size, pp_size

def convert_megatron_checkpoints_to_hfmodes():
    local_path = args.local_dir
    
    model_ckpt_path = get_model_checkpoint_path(local_path)
    hf_model_ckpt_path = get_hf_model_checkpoint_path(local_path)
    sharded_dirs, tp_size, pp_size = check_megatron_checkpoint_path(model_ckpt_path)
    mp_size = len(sharded_dirs)
    
    model_state_dict_lst = []
    for i in range(pp_size):
        model_state_dict_lst.append([])
        for j in range(tp_size):
            model_state_dict_lst[i].append("")
    
    print(f'sharded_dirs: {sharded_dirs}, tp_size: {tp_size}, pp_size: {pp_size}, mp_size: {mp_size}')
    
    def process_one_shard(shard_dir):
        model_path = os.path.join(model_ckpt_path, shard_dir, "model.pt")
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        tp_rank, pp_rank = get_tp_pp_rank_from_sharded_dir(shard_dir)
        model_state_dict_lst[pp_rank][tp_rank] = state_dict

    # with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
    #     for rank in range(1, mp_size):
    #         executor.submit(process_one_shard, sharded_dirs[rank])
    for sharded_dir in sharded_dirs:
        process_one_shard(sharded_dir)
    
    state_dict = {}
    config = AutoConfig.from_pretrained(args.hf_model_path)
    if args.test:
        ref_state_dict = load_file(os.path.join(args.test_hf_dir, 'model.safetensors'))
    
    def handle_qkv_proj(key, config, tensor, state_dict):
        nonlocal tp_size
        
        hidden_size_per_head = config.hidden_size // config.num_attention_heads

        if config.num_key_value_heads >= tp_size:
            q_size_tp = config.hidden_size // tp_size
            kv_size_tp = hidden_size_per_head * config.num_key_value_heads // tp_size
            total_size = q_size_tp + 2 * kv_size_tp
            q_part = tensor[:q_size_tp]
            k_part = tensor[q_size_tp:q_size_tp + kv_size_tp]
            v_part = tensor[q_size_tp + kv_size_tp:total_size]
        else:
            q_size_tp = config.hidden_size // tp_size
            kv_size_tp = hidden_size_per_head
            total_size = q_size_tp + 2 * kv_size_tp
            q_part = tensor[:q_size_tp]
            k_part = tensor[q_size_tp:q_size_tp + kv_size_tp]
            v_part = tensor[q_size_tp + kv_size_tp:total_size]
        
        preffix = '.'.join(key.split('.')[:4])
        suffix = '.'.join(key.split('.')[5:])
        if state_dict.get(f'{preffix}.q_proj.{suffix}') is None:
            state_dict[f'{preffix}.q_proj.{suffix}'] = q_part
        else:
            state_dict[f'{preffix}.q_proj.{suffix}'] = torch.concat([state_dict[f'{preffix}.q_proj.{suffix}'], q_part], dim=0)
        if state_dict.get(f'{preffix}.k_proj.{suffix}') is None:
            state_dict[f'{preffix}.k_proj.{suffix}'] = k_part
        else:
            state_dict[f'{preffix}.k_proj.{suffix}'] = torch.concat([state_dict[f'{preffix}.k_proj.{suffix}'], k_part], dim=0)
        if state_dict.get(f'{preffix}.v_proj.{suffix}') is None:
            state_dict[f'{preffix}.v_proj.{suffix}'] = v_part
        else:
            state_dict[f'{preffix}.v_proj.{suffix}'] = torch.concat([state_dict[f'{preffix}.v_proj.{suffix}'], v_part], dim=0)
            
        return state_dict

    def handle_gate_up_proj(key, config, tensor, state_dict):
        nonlocal tp_size
        
        intermediate_size_tp = config.intermediate_size // tp_size
        gate_weight_tp = tensor[:intermediate_size_tp]
        up_weight_tp = tensor[intermediate_size_tp:]
        preffix = '.'.join(key.split('.')[:4])
        suffix = '.'.join(key.split('.')[5:])
        if state_dict.get(f'{preffix}.gate_proj.{suffix}') is None:
            state_dict[f'{preffix}.gate_proj.{suffix}'] = gate_weight_tp
        else:
            state_dict[f'{preffix}.gate_proj.{suffix}'] = torch.concat([state_dict[f'{preffix}.gate_proj.{suffix}'], gate_weight_tp], dim=0)
        if state_dict.get(f'{preffix}.up_proj.{suffix}') is None:
            state_dict[f'{preffix}.up_proj.{suffix}'] = up_weight_tp
        else:
            state_dict[f'{preffix}.up_proj.{suffix}'] = torch.concat([state_dict[f'{preffix}.up_proj.{suffix}'], up_weight_tp], dim=0)
        
        return state_dict
    
    def merge_between_tp_rank(key, model_state_dict):
        nonlocal state_dict
        
        try:
            tensor = model_state_dict.pop(key)
        except:
            raise RuntimeError(f"error pop: {key}")
        # Embedding layer
        if "model.embed_tokens.weight" in key:
            if state_dict[key] is None:
                state_dict[key] = tensor
            else:
                state_dict[key] = torch.concat([state_dict[key], tensor], dim=0)
            return state_dict
        # Tranformer Layers
        if "input_layernorm.weight" in key:
            if state_dict[key] is None:
                state_dict[key] = tensor
            return state_dict
        if re.search(r"self_attn\.qkv_proj", key):
            state_dict = handle_qkv_proj(key, config, tensor, state_dict)
            return state_dict
        if "self_attn.o_proj.weight" in key:
            if state_dict[key] is None:
                state_dict[key] = tensor
            else:
                state_dict[key] = torch.concat([state_dict[key], tensor], dim=1)
            return state_dict
        if "post_attention_layernorm.weight" in key:
            if state_dict[key] is None:
                state_dict[key] = tensor
            return state_dict
        if re.search(r"mlp\.gate_up_proj\.weight", key):
            state_dict = handle_gate_up_proj(key, config, tensor, state_dict)
            return state_dict
        if "mlp.down_proj.weight" in key:
            if state_dict[key] is None:
                state_dict[key] = tensor
            else:
                state_dict[key] = torch.concat([state_dict[key], tensor], dim=1)
            return state_dict
        # Final LayerNorm
        if "model.norm.weight" in key:
            if state_dict[key] is None:
                state_dict[key] = tensor
            return state_dict
        if not args.tie_word_embedding:
            if args.is_value_model:
                if "lm_head.weight" in key:
                    if state_dict[key] is None:
                        state_dict[key] = tensor
                if "reward_head.weight" in key:
                    if state_dict[key] is None:
                        state_dict[key] = tensor
            else:
                if "lm_head.weight" in key:
                    if state_dict[key] is None:
                        state_dict[key] = tensor
                    else:
                        state_dict[key] = torch.concat([state_dict[key], tensor], dim=0)
            return state_dict
        return state_dict

    for pp_rank in range(pp_size):
        print(f'pp_rank: {pp_rank}')
        for vpp_rank, state_dict_single_layer in enumerate(model_state_dict_lst[pp_rank][0]):
            state_dict_single_layer_iter = state_dict_single_layer.copy()
            keys = state_dict_single_layer_iter.keys()
            for key in keys:
                if "extra_state" in key:
                    continue
                if args.tie_word_embedding and ("lm_head" in key or "reward_head" in key):
                    print(f'skip lm_head and reward_head loading because of tie_word_embeddings')
                    continue
                if re.search(r"self_attn\.qkv_proj", key) is None and re.search(r"gate_up_proj", key) is None:
                    state_dict[key] = None
                for tp_rank in range(tp_size):
                    model_state_dict = model_state_dict_lst[pp_rank][tp_rank][vpp_rank]
                    state_dict = merge_between_tp_rank(key, model_state_dict)
    
    del model_state_dict_lst
    if args.test:
        for key, value in state_dict.items():
            print(key)
            if key not in ref_state_dict:
                raise RuntimeError(f'key: {key} not exist in ref_state_dict {value}')
            if value.shape != ref_state_dict[key].shape:
                raise RuntimeError(f'key: {key} shape mismatch {value.shape}, {ref_state_dict[key].shape}')
            assert value.dtype == ref_state_dict[key].dtype, f'{key} state_dict[key].dtype: {value.dtype} != ref_state_dict[key].dtype: {ref_state_dict[key].dtype}'
            torch.testing.assert_close(value, ref_state_dict[key], atol=1e-4, rtol=1e-4)
        for key in ref_state_dict:
            if key not in state_dict:
                raise RuntimeError(f'key: {key} not exist in state_dict {ref_state_dict[key]}')
        
    
    print('Writing to local disk')
    if args.target_dir is None:
        hf_path = os.path.join(args.local_dir, 'huggingface')
    else:
        hf_path = args.target_dir

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
        upload_model_to_huggingface(hf_path)
    
if __name__ == '__main__':
    if args.backend == "fsdp":
        convert_fsdp_checkpoints_to_hfmodels()
    elif args.backend == "megatron":
        convert_megatron_checkpoints_to_hfmodes()
    else:
        raise NotImplementedError(f"{args.backend} not supported")
