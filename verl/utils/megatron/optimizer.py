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

from megatron.core.optimizer import ChainedOptimizer, DistributedOptimizer, OptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from verl.models.mcore.patch_v012 import PatchedChainedOptimizer, PatchedDistributedOptimizer


def _patch_mcore_v012_optimizers(base_optimizer):
    """Patch the optimizer to support saving optimizer state without DP gathering to memory."""
    if type(base_optimizer) is ChainedOptimizer:
        patch_fn = lambda opt: PatchedDistributedOptimizer(opt) if type(opt) is DistributedOptimizer else opt
        patched_optimizers = [patch_fn(optimizer) for optimizer in base_optimizer.chained_optimizers]
        return PatchedChainedOptimizer(patched_optimizers)

    if type(base_optimizer) is DistributedOptimizer:
        return PatchedDistributedOptimizer(base_optimizer)

    return base_optimizer


def get_megatron_optimizer(
    model,
    config: OptimizerConfig,
    no_weight_decay_cond=None,
    scale_lr_cond=None,
    lr_mult=1.0,
):
    # Base optimizer.
    base_optimizer = get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
        no_weight_decay_cond=no_weight_decay_cond,
        scale_lr_cond=scale_lr_cond,
        lr_mult=lr_mult,
    )

    return _patch_mcore_v012_optimizers(base_optimizer)


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config,
):
    """
    Get the optimizer parameter scheduler for Megatron.
    """
    if config.get("lr_decay_steps", None) is None:
        config.lr_decay_steps = config.total_training_steps
    wsd_decay_steps = None
    if config.get("lr_wsd_decay_steps", None) is not None:
        wsd_decay_steps = config.lr_wsd_decay_steps
    if config.get("lr_warmup_steps_ratio", None) is not None and (config.get("lr_warmup_steps", None) is None or config.lr_warmup_steps <= 0):
        config.lr_warmup_steps = int(config.lr_warmup_steps_ratio * config.lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=config.lr_warmup_init,
        max_lr=config.lr,
        min_lr=config.min_lr,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_decay_steps=config.lr_decay_steps,
        lr_decay_style=config.lr_decay_style,
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=config.total_training_steps,
        wd_incr_style=config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=(not config.use_checkpoint_opt_param_scheduler),
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=config.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def get_megatron_last_lr(optimizer):
    """
    Get the last learning rate from the optimizer parameter scheduler.
    """
    return optimizer.param_groups[0]["lr"]
