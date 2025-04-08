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
    model: torch.nn.Module, optim_dict: Dict[torch.nn.Parameter, torch.optim.Optimizer]
) -> None:
    """
    Register hooks for optimizer step running in backward.

    When fusing the optimizer step into backward, we need to call ``.step()`` on the optimizer
    for a given parameter as soon as its gradient is ready. This utility registers post-accumulate-grad
    hooks on all parameters in the model to achieve this.

    Args:
        model (torch.nn.Module): Model whose parameters will be optimized. Note that currently
            hooks for ALL parameters in the model will be registered.
        optim_dict (Dict[torch.nn.Parameter, torch.optim.Optimizer]): Mapping from
            parameters to optimizers.
    """

    def optim_step(param) -> None:
        optim_dict[param].step()
        optim_dict[param].zero_grad()

    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(optim_step)