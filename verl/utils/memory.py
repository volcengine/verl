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

import torch
from torch.distributed.optim import _apply_optimizer_in_backward

def apply_optimizer_in_backward(
        model: torch.nn.Module,
        optim_config

):
    flat_params = [p for p in model.parameters() if p.requires_grad]
    _apply_optimizer_in_backward(
        torch.optim.AdamW,
        flat_params,
        {
            "lr": optim_config.lr,
            "betas": optim_config.get('betas', (0.9, 0.999)),
            "weight_decay":optim_config.get('weight_decay', 1e-2)
        }
    )
    optim_dict = {
        p: p._in_backward_optimizers[0] for p in model.parameters() if hasattr(p, "_in_backward_optimizers")
    }
    return optim_dict



# def register_optim_in_bwd_hooks(
#     # model: torch.nn.Module,
#     handles,
#     optim_dict: Dict[torch.nn.Parameter, torch.optim.Optimizer],
#     acc_steps: int, # number of microbatches to accumulate,
# ) -> None:
#     """
#     Register backward hooks that only perform an optimizer step after `acc_steps`
#     backward calls on each parameter.
#     """
#     def optim_step(param) -> None:
#         # Get or initialize an accumulation counter on the parameter.
#         # if not hasattr(param, '_accumulation_counter'):
#         #     param._accumulation_counter = 0
#         # param._accumulation_counter += 1

#         # Only update when we've accumulated gradients from all microbatches.
#         # if param._accumulation_counter % acc_steps == 0:
#         #     # print("Autocast enabled before optimizer step:", torch.is_autocast_enabled())
#         #     # with torch.amp.autocast(device_type='cuda', enabled=False):
#         #         # print("Autocast Enabled before optimizer step:", torch.is_autocast_enabled())
#         #     param.data = param.data.float()
#         #     print(f"Param data type: {param.data.dtype}")
#         print("STARTING OPTIMIZER STEP")
#         before = param.data
#         optim_dict[param].step()
#         optim_dict[param].zero_grad()
#         after = param.data
#         print(f"PARAM CHANGED: {before == after}")
#         # Resetting or implicitly allowing counter to roll-over
#             # (optional: you could set param._accumulation_counter = 0)
    
#     # for p in model.parameters():
#     #     if p.requires_grad:
#     #         print(f"P REQUIRES GRAD")
#     #         p.register_post_accumulate_grad_hook(optim_step)
    
#     for handle in handles:
#         fp = handle.flat_param
#         fp.register_post_accumulate_grad_hook(optim_step)

# param_to_optim_hook_handle_map = torch.utils.weak.WeakTensorKeyDictionary()
# param_to_acc_grad_map = torch.utils.weak.WeakTensorKeyDictionary()

# @no_type_check
# def _apply_optimizer_in_backward(
#     optimizer_class: Type[torch.optim.Optimizer],
#     named_params: Iterable[torch.nn.Parameter],
#     optimizer_kwargs: Dict[str, Any],
#     register_hook: bool = True,
# ) -> None:
#     """
#     Upon ``backward()``, the optimizer specified for each parameter will fire after
#     the gradient has been accumulated into the parameter.
#     Note - gradients for these parameters will be set to None after ``backward()``.
#     This means that any other optimizer not specified via `_apply_optimizer_in_backward`
#     over this parameter will be a no-op.
#     Args:
#         optimizer_class: (Type[torch.optim.Optimizer]): Optimizer to apply to parameter
#         params: (Iterator[nn.Parameter]): parameters to apply optimizer state to
#         optimizer_kwargs: (Dict[str, Any]): kwargs to pass to optimizer constructor
#         register_hook: (bool): whether to register a hook that runs the optimizer
#             after gradient for this parameter is accumulated. This is the default
#             way that optimizer in backward is implemented, but specific use cases
#             (such as DDP) may wish to override this to implement custom behavior.
#             (Default = True)
#     Example::
#         params_generator = model.parameters()
#         param_1 = next(params_generator)
#         remainder_params = list(params_generator)
#         apply_optimizer_in_backward(torch.optim.SGD, [param_1], {"lr": .02})
#         apply_optimizer_in_backward(torch.optim.Adam, remainder_params, {"lr": .04})
#         model(...).sum().backward() # after backward, parameters will already
#         # have their registered optimizer(s) applied.
#     """
#     torch._C._log_api_usage_once(
#         "torch.distributed.optim.apply_optimizer_in_backward"
#     )

#     @no_type_check
#     def _apply_optimizer_in_backward_to_param(name, param: torch.nn.Parameter) -> None:
#         # view_as creates a node in autograd graph that allows us access to the
#         # parameter's AccumulateGrad autograd function object. We register a
#         # hook on this object to fire the optimizer when the gradient for
#         # this parameter is ready (has been accumulated into .grad field)
#         if param.view_as(param).grad_fn is None:
#             print(f"PARAM WITH NO GRAD FN: {name}")
#             return
#         # Don't create a new acc_grad if we already have one
#         # i.e. for shared parameters or attaching multiple optimizers to a param.
#         if param not in param_to_acc_grad_map:
#             param_to_acc_grad_map[param] = param.view_as(param).grad_fn.next_functions[0][0]

#         optimizer = optimizer_class([param], **optimizer_kwargs)

#         if not hasattr(param, "_in_backward_optimizers"):
#             param._in_backward_optimizers = []  # type: ignore[attr-defined]
#             # TODO: Remove these attributes once we have a better way of accessing
#             # optimizer classes and kwargs for a parameter.
#             param._optimizer_classes = []  # type: ignore[attr-defined]
#             param._optimizer_kwargs = []  # type: ignore[attr-defined]

#         param._in_backward_optimizers.append(optimizer)  # type: ignore[attr-defined]
#         param._optimizer_classes.append(optimizer_class)  # type: ignore[attr-defined]
#         param._optimizer_kwargs.append(optimizer_kwargs)  # type: ignore[attr-defined]

#         if not register_hook:
#             return

#         def optimizer_hook(*_unused) -> None:
#             for opt in param._in_backward_optimizers:  # type: ignore[attr-defined]
#                 opt.step()

#             param.grad = None
#         print(f"{name} AND ACTUALLY NOT RETURNING BUT REGISTERING HOOKS")
#         handle = param_to_acc_grad_map[param].register_hook(optimizer_hook)  # type: ignore[attr-defined]
#         if param not in param_to_optim_hook_handle_map:
#             param_to_optim_hook_handle_map[param] = []
#         param_to_optim_hook_handle_map[param].append(handle)

#     for name, param in named_params:
#         _apply_optimizer_in_backward_to_param(name, param)
