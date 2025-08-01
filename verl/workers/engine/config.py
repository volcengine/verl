from dataclasses import dataclass, fields
from omegaconf import DictConfig
from verl.utils.config import omega_conf_to_dataclass
from typing import Any

"""
Basic idea:
- unify the config feed into the engine by different workers like Actor/Critic
- no default value
- unmutable
"""

@dataclass(frozen=True)
class BaseConfig:
    def get(self, key: str, default_value: Any = None) -> Any:
        if key in {f.name for f in fields(self)}:
            return getattr(self, key)
        return default_value


@dataclass(frozen=True)
class EngineConfig(BaseConfig):
    """Dataclass for Engine configuration."""
    model: 'ModelConfig'
    optim: 'OptimConfig'
    system: 'SystemConfig'
    checkpoint: 'CheckpointConfig'
    ppo_mini_batch_size: int
    train_micro_batch_size_per_gpu: int
    infer_micro_batch_size_per_gpu: int
    ulysses_sequence_parallel_size: int
    strategy: str
    grad_clip: float
    use_dynamic_bsz: bool
    ppo_max_token_len_per_gpu: int
    forward_max_token_len_per_gpu: int
    rollout_n: int


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    """Dataclass for model configuration."""
    path: str
    tokenizer_path: str
    lora_rank: int
    lora_alpha: int
    target_modules: list[str]
    trust_remote_code: bool
    custom_chat_template: str
    override_config: dict
    use_shm: bool
    enable_gradient_checkpointing: bool
    enable_activation_offload: bool
    use_remove_padding: bool
    external_lib: str
    use_liger: bool
    use_fused_kernels: bool
    fused_kernel_options: dict


@dataclass(frozen=True)
class OptimConfig(BaseConfig):
    """Dataclass for optimizer configuration."""
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    total_training_steps: int
    lr_warmup_steps: int
    lr_warmup_steps_ratio: float
    warmup_style: str
    min_lr_ratio: float
    num_cycles: float



# TODO: use inheritance for different backend
# - FSDPSystemConfig(SystemConfig)
# - MCoreSystemConfig(SystemConfig)



"""
Current offload policy logistic

fsdp:
- actor: force to False
- critic: force to False
- ref: force to be CPUOffload(offload_params=True)
fsdp2:
- actor: depend on offload_policy
- critic: depend on offload_policy
- ref: force to be CPUOffload(pin_memory=True)
"""


                # use_orig_params=fsdp_config.get("use_orig_params", False),
                # forward_prefetch=fsdp_config.get("forward_prefetch", False),

@dataclass(frozen=True)
class SystemConfig(BaseConfig):
    """Dataclass for FSDP system configuration."""
    fsdp_size: int
    param_offload: bool
    optimizer_offload: bool
    wrap_policy: dict
    reshard_after_forward: bool
    model_dtype: str
    mixed_precision: dict
    offload_policy: bool
    forward_prefetch: bool
    use_orig_params: bool


@dataclass(frozen=True)
class CheckpointConfig(BaseConfig):
    save_contents: list[str]
    load_contents: list[str]
    async_save: bool


def get_model_config(config):
    model_config = ModelConfig(
        path=config.path,
        tokenizer_path=config.get("tokenizer_path", None),
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        trust_remote_code=config.trust_remote_code,
        custom_chat_template=config.get("custom_chat_template", None),
        override_config=config.override_config,
        use_shm=config.use_shm,
        enable_gradient_checkpointing=config.enable_gradient_checkpointing,
        enable_activation_offload=config.enable_activation_offload,
        use_remove_padding=config.use_remove_padding,
        external_lib=config.external_lib,
        use_liger=config.use_liger,
        use_fused_kernels=config.use_fused_kernels,
        fused_kernel_options=config.fused_kernel_options,
    )
    return model_config

            # min_lr_ratio = config.optim.get("min_lr_ratio", 0.0)
            # num_cycles = config.optim.get("num_cycles", 0.5)

def get_optim_config(config):
    optim_config = OptimConfig(
        lr=config.lr,
        betas=config.get("betas", (0.9, 0.999)),
        weight_decay=config.weight_decay,
        total_training_steps=config.total_training_steps,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_warmup_steps_ratio=config.lr_warmup_steps_ratio,
        warmup_style=config.get("warmup_style", "constant"),
        min_lr_ratio=config.min_lr_ratio,
        num_cycles=config.num_cycles
    )
    return optim_config


def get_system_config(config):
    system_config = SystemConfig(
        fsdp_size=config.fsdp_size,
        param_offload=config.param_offload,
        optimizer_offload=config.optimizer_offload,
        wrap_policy=config.wrap_policy,
        reshard_after_forward=config.reshard_after_forward,
        model_dtype=config.get("model_dtype", "fp32"),
        mixed_precision=config.get("mixed_precision", None),
        offload_policy=config.offload_policy,
        forward_prefetch=config.forward_prefetch,
        use_orig_params=config.get("use_orig_params", False)
    )
    return system_config


def get_checkpoint_config(config):
    checkpoint_config = CheckpointConfig(
        save_contents=config.save_contents,
        load_contents=config.load_contents,
        async_save=config.async_save
    )
    return checkpoint_config


def get_engine_config(config,
                      model_config,
                      optim_config,
                      system_config,
                      checkpoint_config,
                      rollout_n,
                      infer_micro_batch_size_per_gpu):
    engine_config = EngineConfig(
        model=model_config,
        optim=optim_config,
        system=system_config,
        checkpoint=checkpoint_config,
        ppo_mini_batch_size=config.ppo_mini_batch_size,
        train_micro_batch_size_per_gpu=config.ppo_micro_batch_size_per_gpu,
        infer_micro_batch_size_per_gpu=infer_micro_batch_size_per_gpu,
        ulysses_sequence_parallel_size=config.ulysses_sequence_parallel_size,
        strategy=config.strategy,
        grad_clip=config.grad_clip,
        use_dynamic_bsz=config.use_dynamic_bsz,
        ppo_max_token_len_per_gpu=config.ppo_max_token_len_per_gpu,
        forward_max_token_len_per_gpu=config.get("forward_max_token_len_per_gpu", None),        # no such config for actor
        rollout_n=rollout_n,
    )
    return engine_config


def check_config_for_engine():
    pass