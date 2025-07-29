from dataclasses import dataclass, field
from omegaconf import DictConfig
from verl.utils.config import omega_conf_to_dataclass
from .base import EngineRegistry

@dataclass
class EngineConfig:
    """Dataclass for Engine configuration."""
    model: 'ModelConfig'
    optim: 'OptimConfig'
    checkpoint: 'CheckpointConfig'
    ppo_mini_batch_size: int
    ppo_micro_batch_size: int
    forward_micro_batch_size: int
    ppo_micro_batch_size_per_gpu: int
    forward_micro_batch_size_per_gpu: int
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
    fsdp_config: 'SystemConfig'
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


def normalize_config(config):
    return EngineRegistry.get_engine_cls(config.strategy).normalize_config(config)


def engine_config_from_critic(critic_config: DictConfig) -> EngineConfig:
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
        fsdp_config=SystemConfig(
            fsdp_size=critic_config.model.fsdp_config.fsdp_size,
            param_offload=critic_config.model.fsdp_config.param_offload,
            optimizer_offload=critic_config.model.fsdp_config.optimizer_offload,
            wrap_policy=critic_config.model.fsdp_config.wrap_policy,
            forward_prefetch=critic_config.model.fsdp_config.forward_prefetch,
            reshard_after_forward=critic_config.model.fsdp_config.reshard_after_forward,
            model_dtype=critic_config.model.fsdp_config.model_dtype,
            mixed_precision=critic_config.model.fsdp_config.get("mixed_precision", None),
            offload_policy=critic_config.model.fsdp_config.offload_policy
        ),
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
    ret = EngineConfig(
        model=model_config,
        optim=optim_config,
        ppo_mini_batch_size=critic_config.ppo_mini_batch_size,
        ppo_micro_batch_size=critic_config.ppo_micro_batch_size,
        forward_micro_batch_size=critic_config.forward_micro_batch_size,
        ppo_micro_batch_size_per_gpu=critic_config.ppo_micro_batch_size_per_gpu,
        forward_micro_batch_size_per_gpu=critic_config.forward_micro_batch_size_per_gpu,
        ulysses_sequence_parallel_size=critic_config.ulysses_sequence_parallel_size,
        strategy=critic_config.strategy,
        grad_clip=critic_config.grad_clip,
        use_dynamic_bsz=critic_config.use_dynamic_bsz,
        ppo_max_token_len_per_gpu=critic_config.ppo_max_token_len_per_gpu,
        rollout_n=critic_config.rollout_n,
        checkpoint=critic_config.checkpoint
    )
    return ret
