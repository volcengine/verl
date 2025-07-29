from dataclasses import dataclass, field
from omegaconf import DictConfig
from verl.utils.config import omega_conf_to_dataclass
from .base import EngineRegistry

@dataclass
class EngineConfig:
    """Dataclass for Engine configuration."""
    model: 'ModelConfig'
    optim: 'OptimConfig'
    system: 'SystemConfig'
    checkpoint: 'CheckpointConfig'
    ppo_mini_batch_size: int
    ppo_micro_batch_size: int
    ppo_micro_batch_size_per_gpu: int
    ulysses_sequence_parallel_size: int
    strategy: str
    grad_clip: float
    use_dynamic_bsz: bool
    ppo_max_token_len_per_gpu: int
    rollout_n: int


@dataclass
class ModelConfig:
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


@dataclass
class OptimConfig:
    """Dataclass for optimizer configuration."""
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    total_training_steps: int
    lr_warmup_steps: int
    lr_warmup_steps_ratio: float
    warmup_style: str


# TODO: use inheritance for different backend
# - FSDPSystemConfig(SystemConfig)
# - MCoreSystemConfig(SystemConfig)

@dataclass
class SystemConfig:
    """Dataclass for FSDP system configuration."""
    fsdp_size: int
    param_offload: bool
    optimizer_offload: bool
    wrap_policy: dict
    forward_prefetch: bool
    reshard_after_forward: bool
    model_dtype: str
    mixed_precision: dict
    offload_policy: bool


@dataclass
class CheckpointConfig:
    save_contents: list[str]
    load_contents: list[str]
    async_save: bool


def engine_config_for_critic(critic_config: DictConfig) -> EngineConfig:
    """
    Convert a Hydra config to the FSDPEngineConfig dataclass.

    Args:
        critic_config (DictConfig): The config from CriticWorker.

    Returns:
        FSDPEngineConfig: The converted dataclass.
    """
    print(critic_config)
    model_config = ModelConfig(
        path=critic_config.model.path,
        tokenizer_path=critic_config.model.tokenizer_path,
        lora_rank=critic_config.model.lora_rank,
        lora_alpha=critic_config.model.lora_alpha,
        target_modules=critic_config.model.target_modules,
        trust_remote_code=critic_config.model.trust_remote_code,
        custom_chat_template=critic_config.model.get("custom_chat_template", None),
        override_config=critic_config.model.override_config,
        use_shm=critic_config.model.use_shm,
        enable_gradient_checkpointing=critic_config.model.enable_gradient_checkpointing,
        enable_activation_offload=critic_config.model.enable_activation_offload,
        use_remove_padding=critic_config.model.use_remove_padding,
        external_lib=critic_config.model.external_lib
    )
    optim_config = OptimConfig(
        lr=critic_config.optim.lr,
        betas=critic_config.optim.get("betas", (0.9, 0.999)),
        weight_decay=critic_config.optim.weight_decay,
        total_training_steps=critic_config.optim.total_training_steps,
        lr_warmup_steps=critic_config.optim.lr_warmup_steps,
        lr_warmup_steps_ratio=critic_config.optim.lr_warmup_steps_ratio,
        warmup_style=critic_config.optim.get("warmup_style", "constant")
    )
    system_config = SystemConfig(
        fsdp_size=critic_config.model.fsdp_config.fsdp_size,
        param_offload=critic_config.model.fsdp_config.param_offload,
        optimizer_offload=critic_config.model.fsdp_config.optimizer_offload,
        wrap_policy=critic_config.model.fsdp_config.wrap_policy,
        forward_prefetch=critic_config.model.fsdp_config.forward_prefetch,
        reshard_after_forward=critic_config.model.fsdp_config.reshard_after_forward,
        model_dtype=critic_config.model.fsdp_config.model_dtype,
        mixed_precision=critic_config.model.fsdp_config.get("mixed_precision", None),
        offload_policy=critic_config.model.fsdp_config.offload_policy
    )
    ret = EngineConfig(
        model=model_config,
        optim=optim_config,
        system=system_config,
        ppo_mini_batch_size=critic_config.ppo_mini_batch_size,
        ppo_micro_batch_size=critic_config.ppo_micro_batch_size,
        ppo_micro_batch_size_per_gpu=critic_config.ppo_micro_batch_size_per_gpu,
        ulysses_sequence_parallel_size=critic_config.ulysses_sequence_parallel_size,
        strategy=critic_config.strategy,
        grad_clip=critic_config.grad_clip,
        use_dynamic_bsz=critic_config.use_dynamic_bsz,
        ppo_max_token_len_per_gpu=critic_config.ppo_max_token_len_per_gpu,
        rollout_n=critic_config.rollout_n,
        checkpoint=critic_config.checkpoint
    )
    return ret


def engine_config_for_actor(actor_config: DictConfig) -> EngineConfig:
    """
    Convert a Hydra config to the FSDPEngineConfig dataclass.

    Args:
        actor_config (DictConfig): The config from CriticWorker.

    Returns:
        FSDPEngineConfig: The converted dataclass.
    """
    print(actor_config)
    model_config = ModelConfig(
        path=actor_config.model.path,
        tokenizer_path=None,
        lora_rank=actor_config.model.lora_rank,
        lora_alpha=actor_config.model.lora_alpha,
        target_modules=actor_config.model.target_modules,
        trust_remote_code=actor_config.model.trust_remote_code,
        custom_chat_template=actor_config.model.get("custom_chat_template", None),
        override_config=actor_config.model.override_config,
        use_shm=actor_config.model.use_shm,
        enable_gradient_checkpointing=actor_config.model.enable_gradient_checkpointing,
        enable_activation_offload=actor_config.model.enable_activation_offload,
        use_remove_padding=actor_config.model.use_remove_padding,
        external_lib=actor_config.model.external_lib
    )
    optim_config = OptimConfig(
        lr=actor_config.actor.optim.lr,
        betas=actor_config.actor.optim.get("betas", (0.9, 0.999)),
        weight_decay=actor_config.actor.optim.weight_decay,
        total_training_steps=actor_config.actor.optim.total_training_steps,
        lr_warmup_steps=actor_config.actor.optim.lr_warmup_steps,
        lr_warmup_steps_ratio=actor_config.actor.optim.lr_warmup_steps_ratio,
        warmup_style=actor_config.actor.optim.get("warmup_style", "constant")
    )
    system_config = SystemConfig(
        fsdp_size=actor_config.actor.fsdp_config.fsdp_size,
        param_offload=actor_config.actor.fsdp_config.param_offload,
        optimizer_offload=actor_config.actor.fsdp_config.optimizer_offload,
        wrap_policy=actor_config.actor.fsdp_config.wrap_policy,
        forward_prefetch=actor_config.actor.fsdp_config.forward_prefetch,
        reshard_after_forward=actor_config.actor.fsdp_config.reshard_after_forward,
        model_dtype=actor_config.actor.fsdp_config.get("model_dtype", "fp32"),
        mixed_precision=actor_config.actor.fsdp_config.get("mixed_precision", None),
        offload_policy=actor_config.actor.fsdp_config.offload_policy
    )
    ret = EngineConfig(
        model=model_config,
        optim=optim_config,
        system=system_config,
        ppo_mini_batch_size=actor_config.actor.ppo_mini_batch_size,
        ppo_micro_batch_size=actor_config.actor.ppo_micro_batch_size,
        ppo_micro_batch_size_per_gpu=actor_config.actor.ppo_micro_batch_size_per_gpu,
        ulysses_sequence_parallel_size=actor_config.actor.ulysses_sequence_parallel_size,
        strategy=actor_config.actor.strategy,
        grad_clip=actor_config.actor.grad_clip,
        use_dynamic_bsz=actor_config.actor.use_dynamic_bsz,
        ppo_max_token_len_per_gpu=actor_config.actor.ppo_max_token_len_per_gpu,
        rollout_n=actor_config.rollout.n, # different key
        checkpoint=actor_config.actor.checkpoint
    )
    return ret
