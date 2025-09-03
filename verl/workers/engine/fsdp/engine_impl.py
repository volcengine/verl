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
"""
The concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP)
"""

import gc
import itertools
import logging
import os
import warnings
from contextlib import nullcontext
from typing import Callable

import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_torch_device,
    is_cuda_available,
    is_npu_available,
)
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.py_functional import append_to_dict, convert_to_regular_types
from verl.utils.seqlen_balancing import get_reverse_idx
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

from verl.trainer.config import CheckpointConfig
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig

from ..base import BaseEngine, EngineRegistry
from .utils import create_device_mesh, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


@EngineRegistry.register(["fsdp", "fsdp2"])
class FSDPEngine(BaseEngine):
    """
    Concrete Engine implementation using PyTorch FullyShardedDataParallel (FSDP).

    Supports model sharding, activation/optimizer offloading, LoRA, and sequence parallelism.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the FSDPEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.

        Args:
            config: Configuration object with FSDP and model settings.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None

        self.rank = torch.distributed.get_rank()
        # build device mesh for Ulysses Sequence Parallel

        self.use_remove_padding = self.model_config.use_remove_padding

        self._init_device_mesh()

        # set FSDP offload params
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile  #  use torch compile by default
            else entropy_from_logits
        )

    def is_collect(self):
        is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
        return is_collect

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under FSDP.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.model_config.external_lib)

        self._build_model_optimizer()

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
            log_gpu_memory_usage("After offload model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.optimizer)
            log_gpu_memory_usage("After offload optimizer during init", logger=logger)

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_contents=self.checkpoint_config,
        )

    def _init_device_mesh(self):
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.engine_config.fsdp_size

        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_sequence_parallel_size
        dp_size = self.get_data_parallel_size()
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp_size, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

    def _build_module(self):
        from verl.utils.model import get_hf_auto_model_class
        from verl.utils.torch_dtypes import PrecisionType

        torch_dtype = self.engine_config.model_dtype

        if torch_dtype is None:
            # if it is training, we force torch_dtype to fp32
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16

        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not self.model_config.hf_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            auto_class = get_hf_auto_model_class(hf_config=self.model_config.hf_config)

            module = auto_class.from_pretrained(
                pretrained_model_name_or_path=self.model_config.local_path,
                torch_dtype=torch_dtype,
                config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )

            use_liger = self.model_config.use_liger
            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=module)

            fused_kernel_options = self.model_config.fused_kernel_options
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            use_fused_kernels = self.model_config.use_fused_kernels
            apply_monkey_patch(
                model=module,
                use_remove_padding=self.use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype
            module.to(torch_dtype)

            if self.model_config.enable_gradient_checkpointing:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return module

    def _build_lora_module(self, module):
        module.enable_input_require_grads()
        # Convert config to regular Python types before creating PEFT model
        lora_config = {
            "task_type": TaskType.CAUSAL_LM,
            "r": self.model_config.lora_rank,
            "lora_alpha": self.model_config.lora_alpha,
            "target_modules": convert_to_regular_types(self.model_config.target_modules),
            "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
            "bias": "none",
        }
        module = get_peft_model(module, LoraConfig(**lora_config))
        return module

    def _build_fsdp_module(self, module):
        # TODO(ziheng): need to improve
        from torch.distributed.fsdp import CPUOffload, MixedPrecision

        from verl.utils.torch_dtypes import PrecisionType

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=self.engine_config.wrap_policy,
            is_lora=self.model_config.lora_rank > 0,
        )

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload because it causes incorrect results when using grad accumulation
        if self.engine_config.strategy == "fsdp":
            # cpu_offload:
            # - actor: None
            # - critic: None
            # - ref: CPUOffload(offload_params=True)

            # We force reference policy to use CPUOffload to save memory.
            # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
            cpu_offload = None
            if self.engine_config.forward_only:
                cpu_offload = CPUOffload(offload_params=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False

            module = FSDP(
                module,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.engine_config.forward_prefetch,
                use_orig_params=self.engine_config.use_orig_params,
                cpu_offload=cpu_offload,
            )
        elif self.engine_config.strategy == "fsdp2":
            # - actor: offload_policy
            # - critic: offload_policy
            # - ref: CPUOffloadPolicy(pin_memory=True)
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if self.engine_config.offload_policy or self.engine_config.forward_only:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": self.engine_config.reshard_after_forward,
            }
            full_state = module.state_dict()
            apply_fsdp2(module, fsdp_kwargs, self.engine_config)
            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {self.engine_config.strategy}")

        if self.model_config.enable_activation_offload:
            enable_gradient_checkpointing = self.model_config.enable_gradient_checkpointing
            enable_activation_offloading(module, self.engine_config.strategy, enable_gradient_checkpointing)
        return module

    def _build_optimizer(self, module):
        from torch import optim

        optimizer = optim.AdamW(
            module.parameters(),
            lr=self.optimizer_config.lr,
            betas=self.optimizer_config.betas,
            weight_decay=self.optimizer_config.weight_decay,
        )
        return optimizer

    def _build_lr_scheduler(self, optimizer):
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        optim_config = self.optimizer_config

        total_steps = optim_config.get("total_training_steps", 0)
        num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
        warmup_style = optim_config.get("warmup_style", "constant")
        min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
        num_cycles = optim_config.get("num_cycles", 0.5)
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        if warmup_style == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif warmup_style == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")
        return lr_scheduler

    def _build_model_optimizer(self):
        from verl.utils.model import print_model_size

        # Load base model with specified configuration and dtype
        module = self._build_module()
        # Apply LoRA adapters if low-rank adaptation is enabled
        if self._is_lora:
            module = self._build_lora_module(module)

        # Synchronize all distributed processes before proceeding
        torch.distributed.barrier()
        if self.rank == 0:
            print_model_size(module)
        log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)

        # Wrap model with FSDP for distributed training (sharding, mixed precision, etc.)
        log_gpu_memory_usage("Before FSDP", logger=None)
        module = self._build_fsdp_module(module)
        log_gpu_memory_usage("After FSDP", logger=None)

        if not self.engine_config.forward_only:
            # Initialize optimizer with model parameters and config settings
            optimizer = self._build_optimizer(module)
            # Create learning rate scheduler with warmup and decay settings
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_mode(self):
        """
        Return a context manager that switches to training mode with FSDP-specific handling.

        Includes parameter and optimizer offload entry/exit.
        """
        return EngineTrainModeCtx(self)

    def eval_mode(self):
        """
        Return a context manager that switches to evaluation mode with FSDP-specific handling.

        Includes activation offload entry/exit.
        """
        return EngineEvalModeCtx(self)

    def get_data_parallel_rank(self):
        if self.ulysses_device_mesh is not None:
            return self.ulysses_device_mesh["dp"].get_local_rank()
        else:
            return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size

    def prepare_micro_batches(self, data: DataProto):
        """
        Prepare micro batches from data.
        """
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", True)

        # # has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # # select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        # # non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # batch = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # batch.meta_info = copy.deepcopy(data.meta_info)

        if use_dynamic_bsz:
            assert "max_token_len_per_gpu" in data.meta_info, (
                "max_token_len_per_gpu must be set when use_dynamic_bsz is True"
            )
            max_token_len_per_gpu = data.meta_info.get("max_token_len_per_gpu")
            max_token_len = max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batch_size_per_gpu = data.meta_info.get("micro_batch_size_per_gpu")
            micro_batches = data.split(micro_batch_size_per_gpu)
            batch_idx_list = None
        return micro_batches, batch_idx_list

    def forward_backward_batch(self, data: DataProto, loss_function: Callable, forward_only=False) -> list[DataProto]:
        micro_batches, indices = self.prepare_micro_batches(data=data)

        output = []

        ctx = torch.no_grad() if forward_only else nullcontext()

        for micro_batch in micro_batches:
            with ctx:
                # note that loss must be scaled in postprocess_micro_batch_func
                loss, metrics = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    # metrics contain the output, loss is dummy
                    loss.backward()

            output.append(metrics)

        # postprocess and return
        return self.postprocess_batch_func(output, indices, forward_only, data)

    def postprocess_batch_func(self, losses_reduced, indices, forward_only, data: DataProto):
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", True)

        if forward_only:
            # losses_reduced is a list of dict containing outputs for each micro-batch
            # reorder entropy and outputs. Return None for other pp ranks
            # only on last rank. It should be on every tp rank
            output = {}

            for o in losses_reduced:
                for key, val in o.items():
                    if key not in output:
                        output[key] = []
                    output[key].append(val)

            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)

            for key, val in output.items():
                val = torch.cat(val, dim=0)
                if use_dynamic_bsz:
                    assert len(indices) == val.size(0), f"{len(indices)} vs. {val.size()}"
                    val = val[revert_indices]
                output[key] = val

            return output

        else:
            metrics = {}
            # combine metrics of each micro-batch
            metric_micro_batch = losses_reduced
            for metric in metric_micro_batch:
                # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
                append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

            return metrics

    def forward_step(self, micro_batch: DataProto, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """
        Zero gradients and enforce FSDP grad-clipping logic.
        """
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """
        Clip gradients, skip update if non-finite, and step optimizer.

        Returns:
            grad_norm (float): Norm of gradients before clipping.
        """
        assert self.optimizer_config.clip_grad is not None

        if isinstance(self.module, FSDP):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
        elif isinstance(self.module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=self.optimizer_config.clip_grad)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), max_norm=self.optimizer_config.clip_grad
            )

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm

    def lr_scheduler_step(self):
        """
        Advance FSDP scheduler and return updated learning rate.
        """
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]  # only return the first group
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True):
        """
        Move FSDP model and/or optimizer to CPU or GPU with offload support.
        """
        if self.engine_config.forward_only:
            # force cpu_offload
            return

        assert device in ("cuda", "cpu")
        if device == "cuda":
            if not self.engine_config.param_offload:
                if model:
                    load_fsdp_model_to_gpu(self.module)
                if optimizer and self.optimizer is not None:
                    load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if not self.engine_config.param_offload:
                if model:
                    offload_fsdp_model_to_cpu(self.module)
                if optimizer and self.optimizer is not None:
                    offload_fsdp_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """
        Save FSDP checkpoint, handling parameter offload as needed.
        """
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        """
        Load FSDP checkpoint, restoring parameters and optimizer state.
        """
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)


class EngineEvalModeCtx:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        self.engine.mode = "eval"
        if self.engine._is_offload_param:
            load_fsdp_model_to_gpu(self.engine.module)

        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()

        if self.engine._is_offload_param:
            offload_fsdp_model_to_cpu(self.engine.module)
        self.engine.mode = None


class EngineTrainModeCtx:
    def __init__(self, engine: FSDPEngine):
        self.engine = engine

    def __enter__(self):
        self.engine.mode = "train"
        if self.engine._is_offload_param:
            load_fsdp_model_to_gpu(self.engine.module)
        if self.engine._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.engine.optimizer, device_id=get_torch_device().current_device())

        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)
        self.engine.optimizer_zero_grad()

        if self.engine._is_offload_param:
            offload_fsdp_model_to_cpu(self.engine.module)
        if self.engine._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.engine.optimizer)
        self.engine.mode = None


class FSDPEngineWithLMHead(FSDPEngine):
    def forward_step(self, micro_batch: DataProto, loss_function, forward_only):
        use_remove_padding = micro_batch.meta_info.get("use_remove_padding", True)
        use_fused_kernels = micro_batch.meta_info.get("use_fused_kernels", False)
        temperature = micro_batch.meta_info["temperature"]
        calculate_entropy = micro_batch.meta_info.get("calculate_entropy", False)

        device_name = get_device_name()
        # actually, we should avoid assigning like this...
        micro_batch = micro_batch.to(get_device_id())
        micro_batch_tensor = micro_batch.batch.to(device_name)

        response_length = micro_batch_tensor["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch_tensor.keys():
            if "image_bound" in micro_batch_tensor["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch_tensor["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch_tensor["multi_modal_inputs"]]
            else:
                for key in micro_batch_tensor["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch_tensor["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch_tensor["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch_tensor["attention_mask"]
            position_ids = micro_batch_tensor["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch_tensor.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.engine_config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch_tensor["responses"])
                    if calculate_entropy:
                        if not self.engine_config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            output = {"log_probs": log_probs}
            if calculate_entropy:
                output["entropy"] = entropy

            if forward_only:
                return None, output
            else:
                policy_loss, metrics = loss_function(model_output=output, data=micro_batch_tensor)
                return policy_loss, metrics
