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
"""This file contains memory optimizations"""

from typing import Dict 

import torch

def register_optim_in_bwd_hooks(
    model: torch.nn.Module,
    optim_dict: Dict[torch.nn.Parameter, torch.optim.Optimizer],
    acc_steps: int, # number of microbatches to accumulate,
) -> None:
    """
    Register backward hooks that only perform an optimizer step after `acc_steps`
    backward calls on each parameter.
    """
    def optim_step(param) -> None:
        # Get or initialize an accumulation counter on the parameter.
        if not hasattr(param, '_accumulation_counter'):
            param._accumulation_counter = 0
        param._accumulation_counter += 1

        # Only update when we've accumulated gradients from all microbatches.
        if param._accumulation_counter % acc_steps == 0:
            # print("Autocast enabled before optimizer step:", torch.is_autocast_enabled())
            # with torch.amp.autocast(device_type='cuda', enabled=False):
                # print("Autocast Enabled before optimizer step:", torch.is_autocast_enabled())
            param.data = param.data.float()
            print(f"Param data type: {param.data.dtype}")
            optim_dict[param].step()
            # optim_dict[param].zero_grad()
            # Resetting or implicitly allowing counter to roll-over
            # (optional: you could set param._accumulation_counter = 0)
    
    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(optim_step)
