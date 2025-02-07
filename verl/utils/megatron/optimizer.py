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


def _init_distributed_optimizer(self, optimizer, clip_grad, log_num_zeros_in_grad,
                               check_for_nan_in_grad, params_have_main_grad, fp16,
                               bf16, params_dtype, grad_scaler, models, overlap_param_gather: bool):
        """Megatron optimizer initialized WITHOUT the dependency of **get_args()** APIs.

        See top of class definition for argument descriptions.

        The steps in this method create the core mapping between DDP grad
        buffers, parameters, and parameter shard ranges, that is needed for
        converting between model param indexes and main parameter shard
        indexes. This method also updates the optimizer parameter groups
        with the newly created shards.
        """
        import torch
        from megatron import get_args
        from megatron import get_timers
        from megatron import print_rank_0
        from megatron.core import mpu, tensor_parallel

        from megatron.optimizer.optimizer import MixedPrecisionOptimizer, _zero_grad_group_helper
        from megatron.optimizer.utils import shard_buffer

        super(DistributedOptimizer, self).__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            check_for_nan_in_grad, params_have_main_grad,
            fp16, bf16, params_dtype, grad_scaler, models)

        assert isinstance(optimizer, Adam), \
            "Only Adam currently supported, due to checkpointing requirements."

        # Model grad buffer ranges.
        self.model_gbuf_ranges = []
        self.per_bucket_numel = []
        for _, model_chunk in enumerate(self.models):
            self.per_bucket_numel.append(
                {dtype: [bucket.data.numel() for bucket in model_chunk.grad_buffers[dtype].buckets]
                 for dtype in model_chunk.grad_buffers})
            self.model_gbuf_ranges.append(self.build_model_gbuf_range_map(model_chunk))
        self.model_param_gbuf_map = \
            self.build_model_param_gbuf_map(self.model_gbuf_ranges)

        # Optimizer ranges.
        self.model_param_group_index_map, self.opt_group_ranges = \
            self.build_optimizer_group_ranges(self.optimizer.param_groups,
                                              self.model_gbuf_ranges)

        # Allocate main param shards.
        (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,
            self.shard_fp32_groups,
            self.shard_fp32_from_float16_groups,
        ) = self.build_model_and_main_param_groups(self.model_gbuf_ranges,
                                                   self.model_param_gbuf_map,
                                                   self.opt_group_ranges)

        # Initialize param buffers.
        # - These are views on the DDP model's grad buffers, that share
        #   storage & have their own dtype. This is safe because the param
        #   dtype size is always <= grad dtype size.
        self.param_buffers = []
        for model_index, model in enumerate(self.models):
            current_param_buffers = {}
            for dtype, grad_buffer in model.grad_buffers.items():
                size_ratio = torch.finfo(dtype).bits // torch.finfo(params_dtype).bits
                current_param_buffers[dtype] = []
                for bucket in grad_buffer.buckets:

                    # Handle older/newer method for getting untyped storage.
                    try:
                        storage = bucket.data.storage()._untyped()
                    except:
                        storage = bucket.data.storage().untyped()

                    # Typed param buffer.
                    param_buffer = torch.tensor(
                        storage,
                        dtype = params_dtype,
                        device = bucket.data.device)

                    # .storage() ignores views / slices, so param_buffer now points to the start
                    # of the grad_buffer instead of to the start of each bucket. As a result,
                    # add bucket.offset to make sure param_buffers point to the right region of
                    # memory.
                    # Since we want the start of each bucket's param_buffer to coincide with the
                    # start of the same bucket's grad_buffer (this ensures that zeroing the grad
                    # buffer does not zero out params in the param_buffer before they are copied
                    # into the model_params), multiply the offset by the size ratio of grads and
                    # params.
                    offset = bucket.offset * size_ratio
                    param_buffer = param_buffer[offset:offset+bucket.data.numel()]
                    assert param_buffer.data_ptr() == bucket.data.data_ptr(), \
                        "param_buffer and grad_buffer for same bucket should start at the same byte address"
                    assert param_buffer.numel() == bucket.data.numel(), \
                        "param_buffer and grad_buffer for same bucket should have the same number of elements"
                    current_param_buffers[dtype].append(param_buffer)
            self.param_buffers.append(current_param_buffers)

        # Now construct data structures to manage all-gather handles.
        self.all_gather_handles = []
        self.all_gather_handle_index_to_bucket_index_map = []
        self.model_index_to_all_gather_handle_index_map = {}
        self.param_to_all_gather_handle_index_map = {}
        self.param_buffer_copied = []

        self.pbuf_view_items = self.get_model_param_buffer_dp_views()
        for (model_index, dtype, bucket_index, _, _) in self.pbuf_view_items:
            self.all_gather_handle_index_to_bucket_index_map.append((model_index, dtype, bucket_index))
            all_gather_handle_index = len(self.all_gather_handle_index_to_bucket_index_map) - 1

            # Store all all_gather_handle_indices relevant to a particular model chunk.
            if model_index not in self.model_index_to_all_gather_handle_index_map:
                self.model_index_to_all_gather_handle_index_map[model_index] = []
            self.model_index_to_all_gather_handle_index_map[model_index].append(all_gather_handle_index)

            for param in self.models[model_index].grad_buffers[dtype].buckets[bucket_index].params_list:
                self.param_to_all_gather_handle_index_map[param] = all_gather_handle_index
            self.param_buffer_copied.append(False)
        self.num_all_gather_handles = len(self.all_gather_handle_index_to_bucket_index_map)

        self.overlap_param_gather = overlap_param_gather
        if self.overlap_param_gather:
            self.remove_pre_hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(
                self._make_forward_pre_hook())
        else:
            self.remove_pre_hook_handle = None

        self.update_successful = False

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        self.optimizer.param_groups = \
            [ g["orig_group"] for g in self.opt_group_ranges ]
        self.optimizer.load_state_dict(self.optimizer.state_dict())


DistributedOptimizer.__init__ = _init_distributed_optimizer