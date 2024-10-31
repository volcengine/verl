# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD
from megatron.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.optimizer.grad_scaler import ConstantGradScaler, DynamicGradScaler
from megatron.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from megatron.optimizer import get_param_groups

from verl.utils.megatron.optimizer_config import OptimizerConfig


def get_megatron_optimizer(
        model,
        config: OptimizerConfig,
        no_weight_decay_cond=None,
        scale_lr_cond=None,
        lr_mult=1.0,
        check_for_nan_in_loss_and_grad=False,
        overlap_param_gather=False  # add for verl
):
    # Base optimizer.
    param_groups = get_param_groups(model, no_weight_decay_cond, scale_lr_cond, lr_mult)

    if config.optimizer == 'adam':
        optimizer = Adam(param_groups,
                         lr=config.lr,
                         weight_decay=config.weight_decay,
                         betas=(config.adam_beta1, config.adam_beta2),
                         eps=config.adam_eps)
    elif config.optimizer == 'sgd':
        optimizer = SGD(param_groups, lr=config.lr, weight_decay=config.weight_decay, momentum=config.sgd_momentum)
    else:
        raise Exception('{} optimizer is not supported.'.format(config.optimizer))

    # Determine whether the params have main-grad field.
    params_have_main_grad = True

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if config.loss_scale:
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:
                grad_scaler = DynamicGradScaler(initial_scale=config.initial_loss_scale,
                                                min_scale=config.min_loss_scale,
                                                growth_factor=2.0,
                                                backoff_factor=0.5,
                                                growth_interval=config.loss_scale_window,
                                                hysteresis=config.hysteresis)

        # Megatron optimizer.
        if config.use_distributed_optimizer:
            return DistributedOptimizer(optimizer, config.clip_grad, config.log_num_zeros_in_grad,
                                        check_for_nan_in_loss_and_grad, params_have_main_grad, config.fp16, config.bf16,
                                        config.params_dtype, grad_scaler, model, overlap_param_gather)
        else:
            return Float16OptimizerWithFloat16Params(optimizer, config.clip_grad, config.log_num_zeros_in_grad,
                                                     check_for_nan_in_loss_and_grad, params_have_main_grad, config.fp16,
                                                     config.bf16, config.params_dtype, grad_scaler, model)

    # FP32.
    return FP32Optimizer(optimizer, config.clip_grad, config.log_num_zeros_in_grad, check_for_nan_in_loss_and_grad,
                         params_have_main_grad, model)
