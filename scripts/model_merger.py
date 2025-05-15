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

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from torch.distributed._tensor import Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoTokenizer,
    GenerationConfig,
)

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, required=True, help="The backend of the model", choices=["fsdp", "megatron"])
parser.add_argument("--tie-word-embedding", action="store_true", help="Whether to tie word embedding weights")
parser.add_argument("--is-value-model", action="store_true", help="Whether the model loaded as value model")
parser.add_argument("--hf_model_path", type=str, required=True, help="The path for the huggingface model")
parser.add_argument(
    "--local_dir",
    type=str,
    required=True,
    help=("The path for your saved model. For megatron, point to the base dir of model, rng, optimizer checkpoints, commonly be `config.default_local_dir/global_step_\{global_step\}`."),
)
parser.add_argument("--target_dir", required=False, default="tmp", type=str, help="The path for the target model")
parser.add_argument("--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
parser.add_argument("--test", action="store_true", help="test correctness of hf_model")
parser.add_argument(
    "--test_hf_dir",
    type=str,
    required=False,
    help="test correctness of hf_model, , with hf_model in checkpoint.contents",
)
parser.add_argument("--private", required=False, default=False, help="Whether to upload the model to private repo")

args = parser.parse_args()
os.makedirs(args.target_dir, exist_ok=True)
if args.test:
    assert args.test_hf_dir is not None, "You must run verl save checkpoint first, with hf_model in checkpoint.contents, and provide the directory here"


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
    api.create_repo(repo_id=args.hf_upload_path, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=hf_path, repo_id=args.hf_upload_path, repo_type="model")


def test_fsdp_state_dict(
    auto_model_class,
    original_hf_model_path: str,
    collected_state_dict: Dict[str, torch.Tensor],
) -> bool:
    # load original model using bf16 since we collected state_dict with bf16
    original_model = auto_model_class.from_pretrained(original_hf_model_path, torch_dtype=torch.bfloat16)
    original_state_dict = original_model.state_dict()
    del original_model  # Free memory

    original_keys = set(original_state_dict.keys())
    collected_keys = set(collected_state_dict.keys())

    missing_keys = original_keys - collected_keys
    assert len(missing_keys) == 0, f"Missing keys in collected state dict: {list(sorted(missing_keys))}"

    extra_keys = collected_keys - original_keys
    assert len(extra_keys) == 0, f"Extra keys in collected state dict: {list(sorted(extra_keys))}"

    for key in original_keys:
        original_shape = original_state_dict[key].shape
        collected_shape = collected_state_dict[key].shape
        assert original_shape == collected_shape, f"Shape mismatch for key '{key}': original {original_shape} vs collected {collected_shape}"

        original_dtype = original_state_dict[key].dtype
        collected_dtype = collected_state_dict[key].dtype
        assert original_dtype == collected_dtype, f"Dtype mismatch for key '{key}': original {original_dtype} vs collected {collected_dtype}"

        torch.testing.assert_close(original_state_dict[key], collected_state_dict[key], atol=1e-4, rtol=1e-4)

    print("FSDP checks passed: The merged state_dict matches the hf model saved by FSDPCheckpointManager.")
    return True


def patch_model_generation_config(model, hf_model_path):
    """
    The generation_config created from model config may be different to the pretrained model,
    this may lead to error when generating: https://github.com/volcengine/verl/issues/1246

    This function patch the generation_config created from model config to the pretrained model.
    """
    if model.can_generate():
        try:
            model.generation_config = GenerationConfig.from_pretrained(hf_model_path)
        except OSError:
            print(f"Warning: Generation config file not found in {hf_model_path}, using a generation config created from the model config.")
            pass
    return model


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

    state_dict = torch.load(os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt"), map_location="cpu", weights_only=False)
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]

    if isinstance(weight, DTensor):
        # get sharding info
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        # for non-DTensor
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}"

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing model shards with {total_shards} {mesh_shape} in total")

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank, model_state_dict_lst)
    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except Exception:
                print("-" * 30)
                print(model_state_dict)
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == "dp" or mesh_dim_names[0] == "ddp":
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        if key in param_placements:
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
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    hf_path = os.path.join(local_dir, "huggingface") if args.target_dir is None else args.target_dir
    config = AutoConfig.from_pretrained(args.hf_model_path)

    if "ForTokenClassification" in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif "ForCausalLM" in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif "ForConditionalGeneration" in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture {config['architectures']}")

    if args.test:
        print("Running compatibility test")
        test_fsdp_state_dict(auto_model, args.test_hf_dir, state_dict)

    with torch.device("meta"):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model = patch_model_generation_config(model, args.hf_model_path)

    print(f"Saving model to {hf_path}")
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict
    del model

    print("Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
    tokenizer.save_pretrained(hf_path)

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
        assert "model.pt" in os.listdir(os.path.join(model_path, sharded_dir)), f"model.pt not found in {sharded_dir}"
        tp_rank = int(match.group(1))
        pp_rank = int(match.group(2))
        if tp_size < tp_rank + 1:
            tp_size = tp_rank + 1
        if pp_size < pp_rank + 1:
            pp_size = pp_rank + 1
    return sharded_dirs, tp_size, pp_size


def convert_megatron_checkpoints_to_hfmodels():
    from verl.utils.megatron_utils import get_model_checkpoint_path

    local_path = args.local_dir

    model_ckpt_path = get_model_checkpoint_path(local_path)
    sharded_dirs, tp_size, pp_size = check_megatron_checkpoint_path(model_ckpt_path)
    mp_size = len(sharded_dirs)

    model_state_dict_lst = []
    for i in range(pp_size):
        model_state_dict_lst.append([])
        for j in range(tp_size):
            model_state_dict_lst[i].append("")

    print(f"sharded_dirs: {sharded_dirs}, tp_size: {tp_size}, pp_size: {pp_size}, mp_size: {mp_size}")

    def process_one_shard(shard_dir, model_state_dict_lst):
        model_path = os.path.join(model_ckpt_path, shard_dir, "model.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        tp_rank, pp_rank = get_tp_pp_rank_from_sharded_dir(shard_dir)
        model_state_dict_lst[pp_rank][tp_rank] = state_dict

    # with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
    #     for rank in range(1, mp_size):
    #         executor.submit(process_one_shard, sharded_dirs[rank])
    for sharded_dir in sharded_dirs:
        process_one_shard(sharded_dir, model_state_dict_lst)

    state_dict = {}
    config = AutoConfig.from_pretrained(args.hf_model_path)
    if args.test:
        ref_state_dict = load_file(os.path.join(args.test_hf_dir, "model.safetensors"))

    def merge_across_tp(key, tp_data):
        if "linear_fc1.weight" in key:
            # if the tensor is gate and proj
            gate_lst = []
            up_lst = []
            for infer_param in tp_data:
                gate, up = infer_param.chunk(2)
                gate_lst.append(gate)
                up_lst.append(up)
            gate = torch.cat(gate_lst, dim=0)
            up = torch.cat(up_lst, dim=0)
            tp_data = [gate, up]
        elif "self_attention.linear_qkv." in key and "layer_norm" not in key:
            # if the tensor is qkv, for each param on tp, split into q, k, v
            # concat q, k, v separately.
            q_lst = []
            k_lst = []
            v_lst = []
            assert config.num_attention_heads % config.num_key_value_heads == 0
            num_q_per_kv = config.num_attention_heads // config.num_key_value_heads
            assert tp_data[0].shape[0] % (num_q_per_kv + 2) == 0
            kv_size_per_tp = tp_data[0].shape[0] // (num_q_per_kv + 2)
            split_size = [kv_size_per_tp * num_q_per_kv, kv_size_per_tp, kv_size_per_tp]
            for infer_param in tp_data:
                num_query_groups_per_partition = config.num_key_value_heads // tp_size
                for chunk in infer_param.chunk(num_query_groups_per_partition):
                    split_size = [
                        kv_size_per_tp * num_q_per_kv // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition,
                        kv_size_per_tp // num_query_groups_per_partition,
                    ]
                    q, k, v = chunk.split(split_size)
                    q_lst.append(q)
                    k_lst.append(k)
                    v_lst.append(v)
            q = torch.cat(q_lst, dim=0)
            k = torch.cat(k_lst, dim=0)
            v = torch.cat(v_lst, dim=0)

            tp_data = [q, k, v]

        elif "layer_norm" in key or "layernorm" in key or "output_layer" in key and args.is_value_model:
            tp_data = tp_data[0]
        else:
            dim = 0
            if "linear_fc2.weight" in key or "self_attention.linear_proj" in key:
                dim = 1
            tp_data = torch.cat(tp_data, dim=dim)

        return tp_data

    vpp_size = len(model_state_dict_lst[0][0])
    layers_cum = 0
    for vpp_rank in range(vpp_size):
        for pp_rank in range(pp_size):
            layers_handled = 0
            keys = model_state_dict_lst[pp_rank][0][vpp_rank].keys()
            for key in keys:
                if "extra_state" in key:
                    continue
                if args.tie_word_embedding and ("output_layer" in key):
                    print("skip lm_head and reward_head loading because of tie_word_embeddings")
                    continue
                new_key = key
                if "decoder.layers." in key:
                    local_layer_no = int(key.split(".")[2])
                    layers_handled = max(local_layer_no, layers_handled)
                    global_layer_no = local_layer_no + layers_cum
                    new_key_list = key.split(".")
                    new_key_list[2] = str(global_layer_no)
                    new_key = ".".join(new_key_list)

                tp_data = [model_state_dict_lst[pp_rank][tp_rank][vpp_rank][key] for tp_rank in range(tp_size)]
                merged = merge_across_tp(new_key, tp_data)
                if not isinstance(merged, list):
                    state_dict[new_key] = merged
                elif len(merged) == 3:
                    # split qkv
                    for n, d in zip(["q", "k", "v"], merged):
                        state_dict[new_key.replace("linear_qkv", f"linear_{n}")] = d
                elif len(merged) == 2:
                    # split gate up
                    state_dict[new_key.replace("linear_fc1", "gate_proj")] = merged[0]
                    state_dict[new_key.replace("linear_fc1", "up_proj")] = merged[1]
            layers_cum += layers_handled + 1  # zero based

    del model_state_dict_lst

    params_mapping = [
        # (megatron core gpt model name, vllm model name)
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        ("self_attention.linear_qkv.layer_norm_bias", "input_layernorm.bias"),
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", "self_attn.o_proj"),
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        ("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight"),
        ("mlp.linear_fc1.layer_norm_bias", "post_attention_layernorm.bias"),
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        ("mlp.linear_fc2", "mlp.down_proj"),
        ("decoder.final_layernorm", "model.norm"),
        ("output_layer", "lm_head"),
        ("self_attention.linear_q", "self_attn.q_proj"),
        ("self_attention.linear_k", "self_attn.k_proj"),
        ("self_attention.linear_v", "self_attn.v_proj"),
    ]

    if args.test:
        for original_name, loaded_weight in state_dict.items():
            name = _replace_name(original_name, params_mapping)
            if not name or name.endswith(".bias") and name not in ref_state_dict:
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if args.tie_word_embedding and "lm_head.weight" in name:
                continue
            if name not in ref_state_dict:
                raise RuntimeError(f"key: {name} not exist in state_dict")
            param = ref_state_dict[name]
            assert loaded_weight.dtype == param.dtype
            torch.testing.assert_close(loaded_weight, param, atol=1e-4, rtol=1e-4)

    print("Writing to local disk")
    hf_path = os.path.join(args.local_dir, "huggingface") if args.target_dir is None else args.target_dir

    if "ForTokenClassification" in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif "ForCausalLM" in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif "ForConditionalGeneration" in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture {config['architectures']}")

    with torch.device("meta"):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model = patch_model_generation_config(model, args.hf_model_path)

    print(f"Saving model to {hf_path}")
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict
    del model

    print("Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
    tokenizer.save_pretrained(hf_path)

    if args.hf_upload_path:
        upload_model_to_huggingface(hf_path)


def _replace_name(megatron_name, name_mapping):
    for m_name, v_name in name_mapping:
        if m_name not in megatron_name:
            continue
        if "layers" in megatron_name:  # deal with decoder layers
            megatron_name = megatron_name.replace("decoder", "model")
            megatron_name_list = megatron_name.split(".")
            if "layer_norm_weight" in megatron_name_list or "layer_norm_bias" in megatron_name_list:
                param_name_list = megatron_name_list[:3]
                param_name_list.append(v_name)
                param_name = ".".join(param_name_list)
            else:
                param_name_list = megatron_name_list[:3]
                weight_or_bias = megatron_name_list[-1]
                param_name_list.append(v_name)
                param_name_list.append(weight_or_bias)
                param_name = ".".join(param_name_list)
            return param_name
        else:
            param_name = megatron_name.replace(m_name, v_name)
            return param_name


if __name__ == "__main__":
    if args.backend == "fsdp":
        convert_fsdp_checkpoints_to_hfmodels()
    elif args.backend == "megatron":
        convert_megatron_checkpoints_to_hfmodels()
    else:
        raise NotImplementedError(f"{args.backend} not supported")
