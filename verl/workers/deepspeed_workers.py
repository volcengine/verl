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
Clean DeepSpeed-based Workers for PPO Training (Native DeepSpeed API)

This module provides worker implementations using native DeepSpeed API,
similar to how FSDP workers use native PyTorch FSDP API.
"""

import asyncio
import datetime
import logging
import os
import warnings
from contextlib import nullcontext
from typing import Optional

import psutil
import torch

# Performance optimization: conditionally enable debug logging
_DEBUG_ENABLED = os.getenv("VERL_DEBUG", "0") == "1"


def _get_or_create_event_loop():
    """Return a usable asyncio event loop.

    - If no current loop, create and set a new one (prefer uvloop when available).
    - If the existing loop is closed (e.g., uvloop policy teardown), recreate it.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No current event loop; set policy and create one
        try:
            import uvloop  # type: ignore

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except Exception:
            pass
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop
import torch.distributed
from tensordict import TensorDict
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.third_party.vllm import vllm_version
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.checkpoint.deepspeed_checkpoint_manager import DeepSpeedCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.deepspeed_utils import (
    get_deepspeed_config,
    initialize_deepspeed_engine,
    load_deepspeed_model_to_gpu,
    offload_deepspeed_model_to_cpu,
)
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import (
    compute_position_id_with_mask,
    convert_weight_keys,
    get_generation_config,
    load_valuehead_model,
    print_model_size,
    update_model_config,
)
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.profiler.performance import reduce_timing, simple_timer, topk_reduce_ratio_min_max
from verl.utils.py_functional import append_to_dict, convert_to_regular_types
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.workers.config import DeepSpeedCriticConfig, DeepSpeedEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.actor import DataParallelPPOActor
from verl.workers.critic import DataParallelPPOCritic
from verl.workers.rollout import get_rollout_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def _parse_mixed_precision_config(mixed_precision):
    """Parse mixed_precision config to determine fp16/bf16 flags.

    Args:
        mixed_precision: Can be None, str ("fp16"/"bf16"), or dict {"param_dtype": "bf16", ...}

    Returns:
        tuple: (fp16_enabled, bf16_enabled)
    """
    if mixed_precision is None:
        return False, False
    elif isinstance(mixed_precision, str):
        return mixed_precision == "fp16", mixed_precision == "bf16"
    elif isinstance(mixed_precision, dict):
        param_dtype = mixed_precision.get("param_dtype", "fp32")
        return param_dtype == "fp16", param_dtype == "bf16"
    else:
        return False, False


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    Clean DeepSpeed-based worker using native DeepSpeed API.

    Similar to FSDP worker structure but uses DeepSpeed for training.
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)

        self.config = config

        rollout_cfg = self.config.get("rollout", {}) if isinstance(self.config, DictConfig) else {}
        self._skip_rollout = rollout_cfg.get("skip_rollout", False)
        load_format = rollout_cfg.get("load_format", "")
        self._dummy_rollout = isinstance(load_format, str) and load_format.startswith("dummy")

        # Initialize distributed environment
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # Parse role
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self._is_actor:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)
        if self._is_ref:
            self._register_dispatch_collect_info("ref", dp_rank=self.rank, is_collect=True)

        # Setup profiler
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(f"Invalid role {self.role}")

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        # Setup offload flags
        self._is_offload_param = False
        if self._is_actor:
            deepspeed_config = self.config.actor.deepspeed_config
            self._is_offload_param = deepspeed_config.get("param_offload", False)
        elif self._is_ref:
            deepspeed_config = self.config.ref.deepspeed_config
            self._is_offload_param = deepspeed_config.get("param_offload", False)

        # Normalize batch size config (similar to FSDP worker)
        if self._is_actor:
            world_size = torch.distributed.get_world_size()
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= world_size
            assert self.config.actor.ppo_mini_batch_size > 0

            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= world_size
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0

        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self._lora_rank > 0

    def _build_model_optimizer(
        self,
        model_path: str,
        deepspeed_config: DeepSpeedEngineConfig,
        optim_config: Optional[dict],
        override_model_config: dict,
        use_remove_padding: bool = False,
        use_fused_kernels: bool = False,
        enable_gradient_checkpointing: bool = False,
        trust_remote_code: bool = False,
        use_liger: bool = False,
        role: str = "actor",
    ):
        """
        Build model and optimizer using native DeepSpeed API.

        Returns:
            tuple: (deepspeed_engine, model, optimizer, lr_scheduler, model_config)
        """
        # Load model config
        actor_model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
        )

        # Override config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)

        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # Determine torch dtype
        torch_dtype = deepspeed_config.get("model_dtype", "fp32")
        if torch_dtype == "fp32":
            torch_dtype = torch.float32
        elif torch_dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Create model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=actor_module)

            # Apply monkey patches
            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=1,  # DeepSpeed doesn't support Ulysses yet
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            actor_module.to(torch_dtype)

            # Gradient checkpointing
            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            # LoRA
            if self._is_lora:
                print("Applying LoRA to actor module")
                actor_module.enable_input_require_grads()
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # Initialize DeepSpeed
        if optim_config is not None and role == "actor":
            # Build DeepSpeed config
            # Parse mixed precision config (supports str or dict)
            fp16_enabled, bf16_enabled = _parse_mixed_precision_config(
                deepspeed_config.get("mixed_precision")
            )

            zero_stage = getattr(self.config.actor, "zero_stage", deepspeed_config.get("zero_stage", 2))

            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            per_rank_mini = self.config.actor.ppo_mini_batch_size
            micro_bsz = self.config.actor.get("ppo_micro_batch_size_per_gpu", 1) or 1
            ds_train_batch_size = max(1, per_rank_mini * world_size)
            ds_grad_accum = max(1, per_rank_mini // micro_bsz)

            ds_config = get_deepspeed_config(
                optimizer_type=optim_config.get("optimizer", "AdamW"),
                train_batch_size=ds_train_batch_size,
                train_micro_batch_size_per_gpu=micro_bsz,
                gradient_accumulation_steps=ds_grad_accum,
                zero_stage=zero_stage,
                lr=optim_config.get("lr", 1e-5),
                betas=optim_config.get("betas", [0.9, 0.999]),
                eps=optim_config.get("eps", 1e-8),
                weight_decay=optim_config.get("weight_decay", 0.01),
                fp16_enabled=fp16_enabled,
                bf16_enabled=bf16_enabled,
                cpu_offload=deepspeed_config.get("param_offload", False),
                offload_optimizer=deepspeed_config.get("optimizer_offload", False),
                gradient_clipping=self.config.actor.get("grad_clip", None),
            )

            # Initialize DeepSpeed engine
            ds_engine, optimizer, _, lr_scheduler = initialize_deepspeed_engine(
                model=actor_module,
                config=ds_config,
                model_parameters=actor_module.parameters(),
            )

            return ds_engine, ds_engine.module, optimizer, lr_scheduler, actor_model_config
        else:
            # No optimizer for ref or rollout
            return None, actor_module, None, None, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        """Build rollout engine (vLLM/SGLang)."""

        # Initialize RNG snapshots (needed even for dummy mode)
        device = get_torch_device()
        self.torch_random_states = device.get_rng_state()
        self.gen_random_states = self.torch_random_states.clone()

        rollout_config = omega_conf_to_dataclass(self.config.rollout, dataclass_type=RolloutConfig)
        model_config = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)

        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)

        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=None
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Register dispatch info so Ray routing knows how to gather rollout outputs
        self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)

        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format

        # Switch to trainer mode for sync rollout
        if rollout_config.mode == "sync" and self._is_actor:
            loop = _get_or_create_event_loop()
            loop.run_until_complete(self.trainer_mode())


    async def rollout_mode(self):
        """Context switch to rollout mode."""
        if self._skip_rollout:
            # When rollout is skipped, no weight syncing is required.
            self.base_sync_done = True
            return

        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_deepspeed_model_to_gpu", logger=logger)
        if self._is_offload_param and self.actor_engine is not None:
            load_deepspeed_model_to_gpu(self.actor_engine)
        log_gpu_memory_usage("After load_deepspeed_model_to_gpu", logger=logger)

        # Get model parameters for rollout - ensure we get full tensors
        if self.actor_engine is not None:
            # DeepSpeed engine - need to handle ZeRO partitioned parameters
            # For ZeRO-2, weights are not partitioned, only optimizer states
            # For ZeRO-3, weights ARE partitioned and need gathering

            # Use deepspeed's method to get full state dict if available
            if hasattr(self.actor_engine, 'get_full_state_dict'):
                params = self.actor_engine.get_full_state_dict()
            else:
                # Fallback: get from module directly
                params = self.actor_engine.module.state_dict()

            # Log weight shapes for debugging
            if self.rank == 0:
                for key in list(params.keys())[:3]:  # Check first 3 weights
                    logger.info(f"DeepSpeed weight {key} shape: {params[key].shape}, dtype: {params[key].dtype}")
        else:
            params = self.actor_module.state_dict()

        # Critical: Convert weight keys to match vLLM expectations (like FSDP does)
        if self.actor_engine is not None:
            params = convert_weight_keys(params, self.actor_engine.module)
        else:
            params = convert_weight_keys(params, self.actor_module)

        log_gpu_memory_usage("Before offload_deepspeed_model_to_cpu", logger=logger)
        if self._is_offload_param and self.actor_engine is not None:
            offload_deepspeed_model_to_cpu(self.actor_engine)
        log_gpu_memory_usage("After offload_deepspeed_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        device = get_device_id()

        # Use generator like FSDP does - avoid eager list creation
        # This may help with memory management and weight format compatibility
        def _yield_params():
            for name, param in params.items():
                tensor = param.to(device, non_blocking=True)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                yield name, tensor

        per_tensor_param = _yield_params()

        # Ensure all transfers and memory operations are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Critical fix for DeepSpeed dummy mode compatibility
        # Unlike FSDP, DeepSpeed's weights in dummy mode cause CUDA errors in vLLM
        # Root cause: vLLM's dummy weight initialization is incompatible with DeepSpeed weight format
        # Solution: Skip weight update in dummy mode, let vLLM use its own dummy weights
        if self._dummy_rollout:
            # Dummy mode: Skip weight update entirely
            logger.info("Dummy mode: Skipping weight update, vLLM will use its own dummy-initialized weights")
            del params, per_tensor_param
            aggressive_empty_cache(force_sync=True)
            # Mark as synced to prevent future update attempts
            self.base_sync_done = True
        else:
            # Normal mode: Update weights as usual (this works for DeepSpeed)
            if self.config.rollout.free_cache_engine:
                await self.rollout.resume(tags=["weights"])
            log_gpu_memory_usage("After resume weights", logger=logger)

            await self.rollout.update_weights(per_tensor_param, base_sync_done=self.base_sync_done)
            log_gpu_memory_usage("After update_weights", logger=logger)
            del params, per_tensor_param
            aggressive_empty_cache(force_sync=True)

            if self.config.rollout.free_cache_engine:
                await self.rollout.resume(tags=["kv_cache"])
            log_gpu_memory_usage("After resume kv_cache", logger=logger)

            # Set base_sync_done to True after first sync
            if not self.base_sync_done:
                self.base_sync_done = True

        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch to trainer mode."""
        if self._skip_rollout:
            return

        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        if self.actor_engine is not None:
            self.actor_engine.module.train()
        else:
            self.actor_module.train()

        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(True)

        # Restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize models and engines."""
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        trust_remote_code = self.config.model.get("trust_remote_code", False)

        # Load local model path
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)

        # Initialize tokenizer/processor (before model creation)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # Load generation config
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        # Build actor model with DeepSpeed
        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                deepspeed_config = omega_conf_to_dataclass(self.config.actor.deepspeed_config)
            else:
                optim_config = None
                deepspeed_config = DeepSpeedEngineConfig()

            (
                self.actor_engine,
                self.actor_module,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                deepspeed_config=deepspeed_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=trust_remote_code,
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
            )

            if self._is_offload_param and self.actor_engine is not None:
                offload_deepspeed_model_to_cpu(self.actor_engine)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

        # Build actor wrapper
        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DeepSpeedPPOActor(config=actor_cfg, actor_module=self.actor_module, engine=self.actor_engine)

        # Build rollout
        if self._is_rollout:
            self._build_rollout(trust_remote_code=trust_remote_code)

        # Build reference policy
        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)

            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            (
                self.ref_engine,
                self.ref_module,
                _,
                _,
                _,
            ) = self._build_model_optimizer(
                model_path=local_path,
                deepspeed_config=omega_conf_to_dataclass(self.config.ref.deepspeed_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=trust_remote_code,
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )

            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module)

        # Create checkpoint manager and flops counter
        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)

            # DeepSpeedCheckpointManager expects worker object (accesses self.engine.engine)
            # Store engine reference for checkpoint manager to access
            self.engine = self.actor_engine
            self.checkpoint_manager = DeepSpeedCheckpointManager(engine=self)

        if not self._is_actor and self._is_rollout:
            # Standalone rollout checkpoint manager (load only)
            # Store engine reference for checkpoint manager to access
            self.engine = self.actor_engine if self.actor_engine is not None else None
            if self.engine is not None:
                self.checkpoint_manager = DeepSpeedCheckpointManager(engine=self)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        """Update actor policy using PPO."""
        assert self._is_actor
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.actor_engine)

        data = data.to("cpu")

        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(data=data)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        lr = self.actor_lr_scheduler.get_last_lr()[0]
        metrics["actor/lr"] = lr
        self.actor_lr_scheduler.step()

        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.actor_engine)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        """Generate sequences using vLLM/SGLang rollout."""
        assert self._is_rollout

        prompts = prompts.to(get_device_id())

        eos_id = (
            self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id
        )
        pad_id = (
            self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id
        )

        if prompts.meta_info is None:
            prompts.meta_info = {}
        prompts.meta_info.setdefault("eos_token_id", eos_id)
        prompts.meta_info.setdefault("pad_token_id", pad_id)

        timing_generate: dict[str, float] = {}

        # Critical fix for dummy mode: Skip actual generation, craft lightweight placeholder outputs
        # This keeps the PPO training stack exercised without invoking vLLM kernels.
        if self._dummy_rollout:
            import time
            # Dummy mode also needs mode switching for RNG consistency
            if self._is_actor:
                import asyncio
                loop = _get_or_create_event_loop()
                loop.run_until_complete(self.rollout_mode())

            start = time.perf_counter()

            idx = prompts.batch["input_ids"]
            batch_size = idx.size(0)
            device = idx.device

            if "attention_mask" in prompts.batch.keys():
                attention_mask = prompts.batch["attention_mask"]
            else:
                attention_mask = torch.ones_like(idx, dtype=torch.int64, device=device)

            if "position_ids" in prompts.batch.keys():
                position_ids = prompts.batch["position_ids"]
            else:
                seq_len = idx.size(-1)
                base_positions = torch.arange(seq_len, device=device, dtype=torch.int64)
                position_ids = base_positions.unsqueeze(0).expand_as(idx)

            eos_token_meta = prompts.meta_info.get("eos_token_id", eos_id)
            if isinstance(eos_token_meta, (list, tuple)):
                eos_token_value = eos_token_meta[0]
            elif isinstance(eos_token_meta, torch.Tensor):
                eos_token_value = eos_token_meta.view(-1)[0].item()
            elif eos_token_meta is None:
                eos_token_value = pad_id
            else:
                eos_token_value = eos_token_meta

            try:
                eos_token_value = int(eos_token_value)
            except (TypeError, ValueError):
                eos_token_value = int(pad_id)

            response_length = 1
            responses = torch.full(
                (batch_size, response_length), eos_token_value, dtype=idx.dtype, device=device
            )
            seq = torch.cat([idx, responses], dim=-1)

            delta_position_id = torch.arange(
                1, response_length + 1, device=position_ids.device, dtype=position_ids.dtype
            )
            delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            if position_ids.dim() == 3:  # e.g. mrope (batch, num_heads, seq_len)
                delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                    batch_size, position_ids.size(1), -1
                )
            response_position_ids = position_ids[..., -1:] + delta_position_id
            extended_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = torch.ones(
                (batch_size, response_length), dtype=attention_mask.dtype, device=attention_mask.device
            )
            extended_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)

            batch = TensorDict(
                {
                    "prompts": idx,
                    "responses": responses,
                    "input_ids": seq,
                    "attention_mask": extended_attention_mask,
                    "position_ids": extended_position_ids,
                },
                batch_size=batch_size,
            )

            if hasattr(prompts.batch, "get") and prompts.batch.get("rollout_log_probs") is not None:
                batch["rollout_log_probs"] = prompts.batch["rollout_log_probs"]

            non_tensor_batch = (
                prompts.non_tensor_batch.copy()
                if isinstance(prompts.non_tensor_batch, dict)
                else dict(prompts.non_tensor_batch or {})
            )
            non_tensor_batch.pop("raw_prompt_ids", None)
            non_tensor_batch.pop("multi_modal_data", None)

            timing_generate["generate_sequences"] = time.perf_counter() - start

            meta_info = dict(prompts.meta_info or {})
            timing_meta = dict(meta_info.get("timing", {}))
            timing_meta.update(timing_generate)
            meta_info["timing"] = timing_meta

            output = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
            logger.info("Dummy mode: Skipped vLLM generation, returned prompts as placeholder responses")

            # Switch back to trainer mode after dummy generation
            if self._is_actor:
                import asyncio
                loop = _get_or_create_event_loop()
                loop.run_until_complete(self.trainer_mode())

            return output

        # Normal mode: Use vLLM for actual generation
        if self._is_actor:
            import asyncio
            loop = _get_or_create_event_loop()
            loop.run_until_complete(self.rollout_mode())

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            import asyncio
            loop = _get_or_create_event_loop()
            loop.run_until_complete(self.trainer_mode())

        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )

        output.meta_info.setdefault("timing", {}).update(timing_generate)
        output = output.to("cpu")
        get_torch_device().empty_cache()

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.actor_engine)

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.disable_adapter() if is_lora else nullcontext()

        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        with adapter_ctx:
            log_probs, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)

        output = DataProto.from_dict(
            tensors={"old_log_probs": log_probs, "entropys": entropys},
            meta_info={"temperature": self.config.rollout.temperature},
        )

        output = output.to("cpu")

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.actor_engine)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            data = DataProto.from_dict(tensors={"ref_log_prob": data.batch["old_log_probs"]})
            return data

        assert self._is_ref

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz

        data = data.to("cpu")
        log_prob, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
        output = DataProto.from_dict(tensors={"ref_log_prob": log_prob})

        return output.to("cpu")


class DeepSpeedPPOActor(DataParallelPPOActor):
    """PPO actor that integrates DeepSpeed optimizer/ZeRO workflow."""

    def __init__(self, config, actor_module, engine):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=engine.optimizer)
        self.deepspeed_engine = engine
        self._use_manual_backward = bool(int(os.getenv("DS_USE_MANUAL_BACKWARD", "0")))
        self._last_grad_layout: list[tuple[str, int]] = []
        base_opt = getattr(engine.optimizer, "optimizer", None)
        if torch.distributed.get_rank() == 0:
            print(
                f"[DEBUG][DS Actor] optimizer={type(engine.optimizer).__name__}, "
                f"base_optimizer={type(base_opt).__name__ if base_opt else 'None'}, "
                f"use_manual_backward={self._use_manual_backward}"
            )

    def _get_grad_accum_steps(self) -> int:
        engine_attr = getattr(self.deepspeed_engine, "gradient_accumulation_steps", None)
        steps = engine_attr() if callable(engine_attr) else engine_attr
        if steps is None:
            steps = max(1, self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu)
        return steps

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self.config.tis_imp_ratio_cap > 0:
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        if self.config.tis_imp_ratio_cap > 0:
            assert "rollout_log_probs" in data.batch.keys(), (
                "Truncated Importance Sampling (TIS) requires `actor_rollout_ref.rollout.calculate_log_probs=True` "
                "and is not currently supported in Server mode (agent loop)."
            )

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                grad_accum_steps = self._get_grad_accum_steps()
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if self._use_manual_backward:
                    self.actor_optimizer.zero_grad()
                else:
                    self.deepspeed_engine.zero_grad()

                for idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    rollout_log_probs = model_inputs.get("rollout_log_probs")
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_log_probs=rollout_log_probs,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                        micro_batch_metrics = {
                            "actor/kl_loss": kl_loss.detach().item(),
                            "actor/kl_coef": self.config.kl_loss_coef,
                        }
                    else:
                        micro_batch_metrics = {}

                    # Let DeepSpeed handle loss scaling and gradient accumulation
                    is_last_micro = idx == len(micro_batches) - 1
                    self.deepspeed_engine.set_gradient_accumulation_boundary(is_last_micro)
                    self.deepspeed_engine.backward(policy_loss, scale_wrt_gas=True)

                    # Collect metrics (loss will be properly scaled for logging)
                    if self.config.use_dynamic_bsz:
                        metric_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        metric_scale_factor = 1.0 / grad_accum_steps

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * metric_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                # Manual gradient clipping (required for bf16 mode)
                # DeepSpeed's gradient_clipping config doesn't work with bf16
                grad_norm_val = torch.nn.utils.clip_grad_norm_(
                    self.actor_module.parameters(),
                    max_norm=self.config.grad_clip,
                    norm_type=2.0
                )
                append_to_dict(metrics, {"actor/grad_norm": float(grad_norm_val)})

                # Step only once after all micro batches
                self.deepspeed_engine.step()

        if not self._use_manual_backward:
            self.deepspeed_engine.zero_grad()
        return metrics


class DeepSpeedPPOCritic(DataParallelPPOCritic):
    """PPO critic that delegates backward/step to a DeepSpeed engine."""

    def __init__(self, config, critic_module, engine):
        super().__init__(config=config, critic_module=critic_module, critic_optimizer=engine.optimizer)
        self.deepspeed_engine = engine
        self._use_manual_backward = bool(int(os.getenv("DS_USE_MANUAL_BACKWARD", "0")))
        if torch.distributed.get_rank() == 0:
            base_opt = getattr(engine.optimizer, "optimizer", None)
            print(
                f"[DEBUG][DS Critic] optimizer={type(engine.optimizer).__name__}, "
                f"base_optimizer={type(base_opt).__name__ if base_opt else 'None'}, "
                f"use_manual_backward={self._use_manual_backward}"
            )

    def _get_grad_accum_steps(self) -> int:
        engine_attr = getattr(self.deepspeed_engine, "gradient_accumulation_steps", None)
        steps = engine_attr() if callable(engine_attr) else engine_attr
        if steps is None:
            steps = max(1, self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu)
        return steps

    def update_critic(self, data: DataProto):
        self.critic_module.train()

        if 'rng_seed' in data.meta_info:
            rng_seed = data.meta_info['rng_seed']
            if torch.distributed.get_rank() == 0:
                print(f"[DS Critic] Setting RNG seed: {rng_seed}")
            torch.manual_seed(rng_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rng_seed)

        metrics = {}

        select_keys = [
            "input_ids",
            "responses",
            "response_mask",
            "attention_mask",
            "position_ids",
            "values",
            "returns",
        ]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                grad_accum_steps = self._get_grad_accum_steps()
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if self._use_manual_backward:
                    self.critic_optimizer.zero_grad()
                else:
                    self.deepspeed_engine.zero_grad()

                for idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    values = model_inputs["values"]
                    returns = model_inputs["returns"]

                    vpreds = self._forward_micro_batch(model_inputs)

                    vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                        vpreds=vpreds,
                        values=values,
                        returns=returns,
                        response_mask=response_mask,
                        cliprange_value=self.config.cliprange_value,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )

                    # Let DeepSpeed handle loss scaling and gradient accumulation
                    is_last_micro = idx == len(micro_batches) - 1
                    self.deepspeed_engine.set_gradient_accumulation_boundary(is_last_micro)
                    self.deepspeed_engine.backward(vf_loss, scale_wrt_gas=True)

                    # Collect metrics (loss will be properly scaled for logging)
                    if self.config.use_dynamic_bsz:
                        metric_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        metric_scale_factor = 1.0 / grad_accum_steps

                    micro_batch_metrics = {
                        "critic/vf_loss": vf_loss.detach().item() * metric_scale_factor,
                        "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                        "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                    }
                    append_to_dict(metrics, micro_batch_metrics)

                # Manual gradient clipping (required for bf16 mode)
                # DeepSpeed's gradient_clipping config doesn't work with bf16
                grad_norm_val = torch.nn.utils.clip_grad_norm_(
                    self.critic_module.parameters(),
                    max_norm=self.config.grad_clip,
                    norm_type=2.0
                )
                append_to_dict(metrics, {"critic/grad_norm": float(grad_norm_val)})

                # Step only once after all micro batches
                self.deepspeed_engine.step()

        if not self._use_manual_backward:
            self.deepspeed_engine.zero_grad()
        return metrics



class CriticWorker(Worker, DistProfilerExtension):
    """Clean DeepSpeed-based Critic Worker."""

    def __init__(self, config: DictConfig | DeepSpeedCriticConfig, **kwargs):
        Worker.__init__(self)

        if isinstance(config, DictConfig):
            critic_config = omega_conf_to_dataclass(config, dataclass_type=DeepSpeedCriticConfig)
        else:
            critic_config = config

        self.config: DeepSpeedCriticConfig = critic_config

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # Setup profiler
        omega_profiler_config = self.config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=None)
        )

        self._register_dispatch_collect_info("critic", dp_rank=self.rank, is_collect=True)

        self._is_offload_param = self.config.deepspeed_config.get("param_offload", False)
        self._lora_rank = getattr(self.config.model, "lora_rank", 0)
        self._is_lora = self._lora_rank > 0

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize critic model with native DeepSpeed."""

        import_external_libs(self.config.model.get("external_lib", None))

        trust_remote_code = getattr(self.config.model, "trust_remote_code", False)
        use_shm = getattr(self.config.model, "use_shm", False)
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)

        tokenizer_path = getattr(self.config.model, "tokenizer_path", None) or self.config.model.path
        tokenizer_local_path = copy_to_local(tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(tokenizer_local_path, trust_remote_code=trust_remote_code)

        custom_template = getattr(self.config.model, "custom_chat_template", None)
        if custom_template is not None:
            if self.processor is not None:
                self.processor.chat_template = custom_template
            elif self.tokenizer is not None:
                self.tokenizer.chat_template = custom_template

        override_config = getattr(self.config.model, "override_config", {}) or {}
        if isinstance(override_config, DictConfig):
            override_config = OmegaConf.to_container(override_config, resolve=True)
        override_config_kwargs = {}
        if self.tokenizer is not None:
            override_config_kwargs.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )
        override_config_kwargs.update(override_config)

        attn_impl = override_config_kwargs.get("attn_implementation", "flash_attention_2")
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_impl,
        )
        update_model_config(critic_model_config, override_config_kwargs=override_config_kwargs)
        critic_model_config.num_labels = 1
        critic_model_config.classifier_dropout = 0.0
        critic_model_config.hidden_dropout = 0.0
        critic_model_config.summary_dropout_prob = 0.0

        torch_dtype = PrecisionType.to_dtype(getattr(self.config.deepspeed_config, "model_dtype", "fp32"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_module = load_valuehead_model(
                local_path=local_path,
                torch_dtype=torch_dtype,
                model_config=critic_model_config,
                trust_remote_code=trust_remote_code,
            )

        use_remove_padding = getattr(self.config.model, "use_remove_padding", False)
        apply_monkey_patch(
            model=critic_module,
            use_remove_padding=use_remove_padding,
            ulysses_sp_size=1,
        )

        critic_module.to(torch_dtype)

        if getattr(self.config.model, "enable_gradient_checkpointing", False):
            critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self._lora_rank,
                "lora_alpha": getattr(self.config.model, "lora_alpha", 16),
                "target_modules": convert_to_regular_types(getattr(self.config.model, "target_modules", None)),
                "bias": "none",
            }
            exclude_modules = getattr(self.config.model, "exclude_modules", None)
            if exclude_modules is not None:
                lora_config["exclude_modules"] = convert_to_regular_types(exclude_modules)
            critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        # Initialize DeepSpeed
        # Parse mixed precision config (supports str or dict)
        fp16_enabled, bf16_enabled = _parse_mixed_precision_config(
            self.config.deepspeed_config.get("mixed_precision")
        )

        zero_stage = getattr(self.config, "zero_stage", self.config.deepspeed_config.get("zero_stage", 2))

        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        per_rank_mini = self.config.ppo_mini_batch_size
        micro_bsz = self.config.get("ppo_micro_batch_size_per_gpu", 1) or 1
        ds_train_batch_size = max(1, per_rank_mini * world_size)
        ds_grad_accum = max(1, per_rank_mini // micro_bsz)

        ds_config = get_deepspeed_config(
            optimizer_type=self.config.optim.get("optimizer", "AdamW"),
            train_batch_size=ds_train_batch_size,
            train_micro_batch_size_per_gpu=micro_bsz,
            gradient_accumulation_steps=ds_grad_accum,
            zero_stage=zero_stage,
            lr=self.config.optim.lr,
            betas=self.config.optim.get("betas", [0.9, 0.999]),
            weight_decay=self.config.optim.get("weight_decay", 0.01),
            fp16_enabled=fp16_enabled,
            bf16_enabled=bf16_enabled,
            cpu_offload=self.config.deepspeed_config.get("param_offload", False),
            offload_optimizer=self.config.deepspeed_config.get("optimizer_offload", False),
            gradient_clipping=self.config.get("grad_clip", None),
        )

        self.critic_engine, optimizer, _, lr_scheduler = initialize_deepspeed_engine(
            model=critic_module,
            config=ds_config,
            model_parameters=critic_module.parameters(),
        )

        self.critic_module = self.critic_engine.module
        self.critic_optimizer = optimizer
        self.critic_lr_scheduler = lr_scheduler

        self.critic = DeepSpeedPPOCritic(
            config=self.config, critic_module=self.critic_module, engine=self.critic_engine
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan", role="critic_compute_values")
    def compute_values(self, data: DataProto):
        """Run critic forward pass to compute value estimates."""
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.critic_engine)

        micro_batch_size = getattr(self.config, "forward_micro_batch_size_per_gpu", None)
        if micro_batch_size is None:
            micro_batch_size = getattr(self.config, "ppo_micro_batch_size_per_gpu", 1)

        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = getattr(
            self.config, "forward_max_token_len_per_gpu", self.config.ppo_max_token_len_per_gpu
        )
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz

        data = data.to("cpu")  # tensors move to device inside critic.compute_values

        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values}).to("cpu")

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.critic_engine)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="blue", role="critic_update")
    def update_critic(self, data: DataProto):
        """Update critic value function."""
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.critic_engine)

        data = data.to("cpu")

        with Timer(name="update_critic", logger=None):
            metrics = self.critic.update_critic(data=data)

        lr = self.critic_lr_scheduler.get_last_lr()[0]
        metrics["critic/lr"] = lr
        self.critic_lr_scheduler.step()

        output = DataProto(meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.critic_engine)

        return output


class RolloutWorker(Worker):
    """Standalone vLLM/SGLang Rollout Worker."""

    def __init__(self, config: DictConfig, **kwargs):
        Worker.__init__(self)
        self.config = config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize rollout engine only."""
        rollout_config = omega_conf_to_dataclass(self.config.rollout, dataclass_type=RolloutConfig)
        model_config = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)

        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=None
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto):
        """Generate sequences."""
        prompts = prompts.to(get_device_id())
        output = self.rollout.generate_sequences(prompts=prompts)
        return output


# Async variants
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    """Async variant of ActorRolloutRefWorker."""
    pass


class RewardModelWorker(Worker, DistProfilerExtension):
    """
    DeepSpeed-based Reward Model Worker.

    Implements reward model inference using DeepSpeed ZeRO for memory efficiency.
    Supports AutoModelForTokenClassification models.
    """

    def __init__(self, config):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # Note: DeepSpeed doesn't support Ulysses SP, so ulysses_sequence_parallel_size is always 1
        self.ulysses_sequence_parallel_size = 1

        # Create training dispatch
        self._register_dispatch_collect_info("reward", dp_rank=self.rank, is_collect=True)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # Normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

        self._is_offload_param = self.config.deepspeed_config.get("param_offload", False)

    def _build_model(self, config):
        """Build reward model with DeepSpeed."""
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        # Handle chat template switching if needed
        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        # Determine torch dtype
        torch_dtype = self.config.deepspeed_config.get("model_dtype", "fp32")
        if torch_dtype == "fp32":
            torch_dtype = torch.float32
        elif torch_dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16  # default to bf16

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=1,  # DeepSpeed doesn't support Ulysses SP
            )

            reward_module.to(torch_dtype)

        # Initialize DeepSpeed for inference (no optimizer)
        # Parse mixed precision config
        fp16_enabled, bf16_enabled = _parse_mixed_precision_config(
            self.config.deepspeed_config.get("mixed_precision")
        )

        zero_stage = getattr(self.config, "zero_stage", self.config.deepspeed_config.get("zero_stage", 2))

        ds_config = get_deepspeed_config(
            optimizer_type="AdamW",  # Not used but required by config generator
            train_batch_size=1,
            train_micro_batch_size_per_gpu=1,
            gradient_accumulation_steps=1,
            zero_stage=zero_stage,
            lr=1e-5,  # Not used but required
            fp16_enabled=fp16_enabled,
            bf16_enabled=bf16_enabled,
            cpu_offload=self.config.deepspeed_config.get("param_offload", False),
            offload_optimizer=False,  # No optimizer for inference
            disable_scheduler=True,  # No scheduler for inference
        )

        # Remove optimizer from config since this is inference only
        if "optimizer" in ds_config:
            del ds_config["optimizer"]

        # Initialize DeepSpeed engine without optimizer
        ds_engine, _, _, _ = initialize_deepspeed_engine(
            model=reward_module,
            config=ds_config,
            model_parameters=None,  # No parameters needed for inference
        )

        return ds_engine

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize reward model."""
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_engine = self._build_model(config=self.config)
        self.reward_module = self.reward_engine.module

    def _forward_micro_batch(self, micro_batch):
        """Forward pass for a single micro batch."""
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
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

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(
                    input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        """Expand sentence-level scores to token-level."""
        batch_size = data.batch.batch_size[0]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        """Switch chat template if input_tokenizer is different from reward model tokenizer."""
        import numpy as np

        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )

            # extract raw prompt
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                print(f"Switch template. chat: {prompt_with_chat_template}")

            prompt_ids = target_tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)

            # pad or truncate
            if len(prompt_ids) < src_max_length:
                prompt_ids = prompt_ids + [target_tokenizer.pad_token_id] * (src_max_length - len(prompt_ids))
                attn_mask = [1] * len(prompt_ids) + [0] * (src_max_length - len(prompt_ids))
            else:
                prompt_ids = prompt_ids[:src_max_length]
                attn_mask = [1] * src_max_length

            rm_input_ids.append(prompt_ids)
            rm_attention_mask.append(attn_mask)

        # convert to tensors
        rm_input_ids = torch.tensor(rm_input_ids, dtype=torch.long, device=data.batch["input_ids"].device)
        rm_attention_mask = torch.tensor(
            rm_attention_mask, dtype=torch.long, device=data.batch["attention_mask"].device
        )

        # update data
        data.batch["input_ids"] = rm_input_ids
        data.batch["attention_mask"] = rm_attention_mask
        # recompute position_ids
        data.batch["position_ids"] = compute_position_id_with_mask(rm_attention_mask)

        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    def compute_rm_score(self, data: DataProto):
        """Compute reward model scores."""
        if self._is_offload_param:
            load_deepspeed_model_to_gpu(self.reward_engine)

        data = data.to("cpu")

        # Switch chat template if needed
        if self._do_switch_chat_template:
            data = self._switch_chat_template(data)

        # Move data to device
        data = data.to(get_device_id())

        # Compute scores for each micro batch
        micro_batch_size = self.config.get("micro_batch_size_per_gpu", 1)
        batch_size = data.batch.batch_size[0]
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        all_scores = []
        for i in range(num_micro_batches):
            start_idx = i * micro_batch_size
            end_idx = min((i + 1) * micro_batch_size, batch_size)

            micro_batch = {
                "input_ids": data.batch["input_ids"][start_idx:end_idx],
                "attention_mask": data.batch["attention_mask"][start_idx:end_idx],
                "position_ids": data.batch["position_ids"][start_idx:end_idx],
            }

            scores = self._forward_micro_batch(micro_batch)
            all_scores.append(scores)

        # Concatenate all scores
        rm_scores = torch.cat(all_scores, dim=0)

        # Expand to token level if needed
        token_level_scores = self._expand_to_token_level(data, rm_scores)

        output = DataProto.from_dict(tensors={"token_level_scores": token_level_scores})

        if self._is_offload_param:
            offload_deepspeed_model_to_cpu(self.reward_engine)

        return output
