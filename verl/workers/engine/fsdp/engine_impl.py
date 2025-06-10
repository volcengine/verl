
import json
import logging
import os
import warnings
from dataclasses import asdict
from typing import Union

import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.debug.performance import _timer, reduce_timing
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from .utils import create_device_mesh, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()




class EngineEvalModeCtx:
    def __init__(self, engine):
        self.engine = engine
    
    def __enter__(self):
        self.engine.mode = "eval"
        if self.engine._is_offload_param:
            load_fsdp_model_to_gpu(self.engine.critic_module)

        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.critic_module.eval()


    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)
        if self.engine._is_offload_param:
            offload_fsdp_model_to_cpu(self.engine.critic_module)
        self.engine.mode = None
        

class EngineTrainModeCtx:
    def __init__(self, engine):
        self.engine = engine
    
    
    def __enter__(self):
        self.engine.mode = "train"
        if self.engine._is_offload_param:
            load_fsdp_model_to_gpu(self.engine.critic_module)
        if self.engine._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.engine.critic_optimizer, device_id=get_torch_device().current_device())

        self.engine.ulysses_sharding_manager.__enter__()
        self.engine.critic_module.train()


    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.ulysses_sharding_manager.__exit__(exc_type, exc_value, traceback)

        if self.engine._is_offload_param:
            offload_fsdp_model_to_cpu(self.engine.critic_module)
        if self.engine._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        self.engine.mode = None



class FSDPEngine(object):
    def __init__(self, config):
        self.config = config
        self.rank = torch.distributed.get_rank()
        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)        
        self._is_lora = self.config.model.get("lora_rank", 0) > 0
        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload


    def init_model_and_optimizer(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        # from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        # self.critic = DataParallelPPOCritic(config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_contents=self.config.checkpoint.contents,
        )
    
    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        from omegaconf import OmegaConf

        override_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification

        critic_model_config = AutoConfig.from_pretrained(local_path, attn_implementation="flash_attention_2", trust_remote_code=config.model.get("trust_remote_code", False))
        critic_model_config.num_labels = 1
        # patch for kimi-vl
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        init_context = get_init_weight_context_manager(use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0

            critic_module = load_valuehead_model(
                local_path,
                torch_dtype,
                critic_model_config,
                config.model.get("trust_remote_code", False),
            )

            use_remove_padding = config.model.get("use_remove_padding", False)

            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.config.model.lora_rank,
                "lora_alpha": self.config.model.lora_alpha,
                "target_modules": convert_to_regular_types(self.config.model.target_modules),
                "bias": "none",
            }
            critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy, is_lora=self.config.model.get("lora_rank", 0) > 0)

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=False,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        warmup_style = config.optim.get("warmup_style", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if warmup_style == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps)
        elif warmup_style == "cosine":
            critic_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler


    def train_mode(self):
        return EngineTrainModeCtx(self)


    def eval_mode(self):
        return EngineEvalModeCtx(self)
    

    def shard_data(self, data):
        return self.ulysses_sharding_manager.preprocess_data(data)


    def unshard_data(self, data):
        return self.ulysses_sharding_manager.postprocess_data(data)


    def forward_backward_step(self, 
                              batch, 
                              ctx=None, 
                              forward_only=False, 
                              preprocess_fn=None, 
                              postprocess_fn=None):
        assert self.mode is not None
        if self.mode == "train":
            assert forward_only == False
        elif self.mode == "eval":
            assert forward_only ==True

        inputs, ctx = self.preprocess_fn(batch, ctx)
        outputs = self.critic_module(**inputs)
        preds, ctx = self.postprocess_fn(outputs, ctx)
        if forward_only:
            return preds, ctx

        loss, ctx = self.loss_fn(batch, preds, ctx)
        loss.backward()
        return preds, loss, ctx


    def optimizer_zero_grad(self):
        self.critic_optimizer.zero_grad()


    def optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm


    def lr_scheduler_step(self):
        self.critic_lr_scheduler.step()
        lr = self.critic_lr_scheduler.get_last_lr()
        return lr


    def set_preprocess_fn(self, preprocess_fn):
        """
        preprocess_fn(data, ctx) -> inputs, ctx
        """
        self.preprocess_fn = preprocess_fn


    def set_postprocess_fn(self, postprocess_fn):
        """
        postprocess_fn(outputs, ctx) -> preds, ctx
        """
        self.postprocess_fn = postprocess_fn
        

    def set_loss_fn(self, loss_fn):
        """
        loss_fn(data, preds, ctx) -> loss, out_ctx
        """
        self.loss_fn = loss_fn

    def to():
        pass