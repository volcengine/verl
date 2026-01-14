# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from verl.utils.device import get_device_id, get_torch_device


@torch.no_grad()
def offload_veomni_model_to_cpu(model, empty_cache: bool = True):
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    assert ps.dp_mode == "fsdp2", "Only support fsdp2 offloading for VeOmni model"

    if ps.ep_enabled:
        parallel_plan = model.get_parallel_plan()
        if parallel_plan is not None:
            experts_map = parallel_plan.get_fsdp_no_shard_info(model)

            if experts_map:
                for module in experts_map.values():
                    module.to("cpu")
    # Offload FSDP parameters (non-expert parameters)
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_veomni_model_to_gpu(model):
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    assert ps.dp_mode == "fsdp2", "Only support fsdp2 offloading for VeOmni model"

    device = get_device_id()
    if ps.ep_enabled:
        parallel_plan = model.get_parallel_plan()
        if parallel_plan is not None:
            experts_map = parallel_plan.get_fsdp_no_shard_info(model)

            if experts_map:
                for module in experts_map.values():
                    module.to(device)

    model.to(device)


@torch.no_grad()
def offload_veomni_optimizer(optimizer):
    optimizers = []
    # Check if this is a MultiOptimizer (for ep and non-ep parameters when ep+fsdp2 is enabled)
    if hasattr(optimizer, "_is_multi_optimizer") and optimizer._is_multi_optimizer:
        optimizers.extend(optimizer.optimizers_dict.values())
    else:
        optimizers.append(optimizer)

    for opt in optimizers:
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_veomni_optimizer(optimizer, device_id):
    optimizers = []
    # Check if this is a MultiOptimizer (for ep and non-ep parameters when ep+fsdp2 is enabled)
    if hasattr(optimizer, "_is_multi_optimizer") and optimizer._is_multi_optimizer:
        optimizers.extend(optimizer.optimizers_dict.values())
    else:
        optimizers.append(optimizer)

    for opt in optimizers:
        if not opt.state:
            continue
        for param_group in opt.param_groups:
            for param in param_group["params"]:
                state = opt.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device_id, non_blocking=True)
