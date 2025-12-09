# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2025 Individual Contributor: Brilliant Hanabi, furunding
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
import torch
from megatron.core import parallel_state as mpu

import verl.utils.megatron.tensor_parallel as tp_utils
from verl.utils.device import get_device_id
from verl.utils.megatron_utils import default_tp_concat_fn, unwrap_model
from verl.utils.model import normalize_model_name


def per_tensor_generator(
    actor_module,
    model_config,
    weight_converter,
    transformer_config,
    layer_name_mapping,
    convert_qkv_gate_up_by_simple_split=True,
):
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    ep_rank = mpu.get_expert_model_parallel_rank()
    etp_rank = mpu.get_expert_tensor_parallel_rank()
    ep_size = mpu.get_expert_model_parallel_world_size()
    etp_size = mpu.get_expert_tensor_parallel_world_size()
    ep_group = mpu.get_expert_model_parallel_group()
    etp_group = mpu.get_expert_tensor_parallel_group()
    vpp_size = len(actor_module)
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_size = torch.distributed.get_world_size(group=tp_group)

    def tensor_generator():
        for scan_vpp_idx in range(vpp_size):
            existing_keys = set()
            model = unwrap_model(actor_module[scan_vpp_idx])
            for name, param in model.named_parameters():
                existing_keys.add(name)
                yield name, param
            # note
            # there is a bug in megatron GPTModel
            # decoder.layers[n].mlp.router.expert_bias" in GPTModel is not registered in named_parameter, but in
            # state_dict(). for now we patch it by adding those keys to extra_keys.
            extra_keys = [x for x in model.state_dict().keys() if "_extra_state" not in x and x not in existing_keys]
            for name in extra_keys:
                yield name, model.state_dict()[name].to(get_device_id())

    def get_tensor_spec(tensor):
        shape = tensor.shape
        dtype = tensor.dtype
        tensor_parallel = getattr(tensor, "tensor_model_parallel", None)
        partition_dim = getattr(tensor, "partition_dim", None)
        tensor_spec = (shape, dtype, tensor_parallel, partition_dim)
        return tensor_spec

    def make_tensor(tensor_spec):
        tensor = torch.empty(size=tensor_spec[0], dtype=tensor_spec[1], device=get_device_id())
        if tensor_spec[2] is not None:
            tensor.tensor_model_parallel = tensor_spec[2]
        if tensor_spec[3] is not None:
            tensor.partition_dim = tensor_spec[3]
        return tensor

    # we need first make all rank get full model information
    meta_info = []
    for scan_vpp_idx in range(vpp_size):
        existing_keys = set()
        model = unwrap_model(actor_module[scan_vpp_idx])
        for idx, (name, param) in enumerate(model.named_parameters()):
            existing_keys.add(name)
            meta_info.append((pp_rank, scan_vpp_idx, idx, name, get_tensor_spec(param)))
        extra_keys = [
            (x, y) for x, y in model.state_dict().items() if "_extra_state" not in x and x not in existing_keys
        ]
        for name, param in extra_keys:
            meta_info.append((pp_rank, scan_vpp_idx, idx, name, get_tensor_spec(param)))

    obj_spec_output = [None] * mpu.get_pipeline_model_parallel_world_size()
    torch.distributed.all_gather_object(
        object_list=obj_spec_output, obj=meta_info, group=mpu.get_pipeline_model_parallel_group()
    )
    layer_list_meta = [item for sublist in obj_spec_output for item in sublist]

    gen_func = tensor_generator()

    # lazy load tensor for full model
    for cur_pp_rank, scan_vpp_idx, idx, name, tensor_spec in layer_list_meta:
        # fp.write(f"DEBUG: ({cur_pp_rank}, {scan_vpp_idx}, {name})\n")
        if model_config.tie_word_embeddings and ("output_layers" in name):
            import warnings

            warnings.warn(
                "Current model sharing word and embedding weights, skip output layer conversion", stacklevel=2
            )
            continue

        cur_name = normalize_model_name(name, cur_pp_rank, scan_vpp_idx, transformer_config)

        if cur_pp_rank == pp_rank:
            _, cur_tensor = next(gen_func)

        else:
            cur_tensor = None

        if pp_rank == 0:
            if cur_tensor is None:
                cur_tensor = make_tensor(tensor_spec)
                torch.distributed.recv(cur_tensor, group=mpu.get_pipeline_model_parallel_group(), group_src=cur_pp_rank)
        else:
            if cur_tensor is None:
                cur_tensor = make_tensor(tensor_spec)
            else:
                torch.distributed.send(cur_tensor, group=mpu.get_pipeline_model_parallel_group(), group_dst=0)

        # (xya): this is a hack to fix the name of the parameters
        while cur_name.startswith("module."):
            cur_name = cur_name[len("module.") :]

        def gather(tensor, gather_list, group, group_dst, group_rank):
            if group_rank == group_dst:
                torch.distributed.gather(tensor, gather_list, group=group, group_dst=group_dst)
            else:
                torch.distributed.gather(tensor, None, group=group, group_dst=group_dst)

        # EP
        if ".mlp.experts.linear_fc" in cur_name and ep_size > 1:
            num_experts = weight_converter.mcore_config.num_moe_experts
            num_experts_per_rank = num_experts // ep_size
            infer_params = [torch.empty_like(cur_tensor) for _ in range(ep_size)]
            gather(cur_tensor, infer_params, group=ep_group, group_dst=0, group_rank=ep_rank)

            name_prefix, local_expert_id = cur_name.split(".weight")
            local_expert_id = int(local_expert_id)
            global_expert_ids = [num_experts_per_rank * _ep_rank + local_expert_id for _ep_rank in range(ep_size)]
            global_expert_names = [f"{name_prefix}.weight{expert_id}" for expert_id in global_expert_ids]

            for name, param in zip(global_expert_names, infer_params, strict=True):
                if etp_size > 1:
                    # gather etp
                    etp_params = [torch.empty_like(param) for _ in range(etp_size)]
                    gather(param, etp_params, group=etp_group, group_dst=0, group_rank=etp_rank)
                    params = etp_params
                else:
                    params = [param]

                merge_params = default_tp_concat_fn(
                    layer_name_mapping,
                    name,
                    cur_tensor,
                    params,
                    model_config,
                    weight_converter.hf_config,
                    convert_qkv_gate_up_by_simple_split,
                )
                if not isinstance(merge_params, list):
                    merge_params = [merge_params]
                converted_names, converted_params = weight_converter.convert_param(name, merge_params)

                yield from zip(converted_names, [param.detach() for param in converted_params], strict=True)

            continue
        # tp all gather
        if tp_utils.is_tensor_parallel_param(cur_tensor):
            # allocate a new tensor with proper size
            if tp_size <= 1:
                infer_params = [cur_tensor]
            else:
                infer_params = [torch.empty_like(cur_tensor) for _ in range(tp_size)]
                gather(cur_tensor, infer_params, tp_group, group_dst=0, group_rank=tp_rank)
            infer_params = default_tp_concat_fn(
                layer_name_mapping,
                cur_name,
                cur_tensor,
                infer_params,
                model_config,
                weight_converter.hf_config,
                convert_qkv_gate_up_by_simple_split,
            )
        else:
            infer_params = cur_tensor

        if not isinstance(infer_params, list):
            infer_params = [infer_params]
        converted_names, converted_params = weight_converter.convert_param(cur_name, infer_params)

        yield from zip(converted_names, [param.detach() for param in converted_params], strict=True)
