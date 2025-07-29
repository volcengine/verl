from dataclasses import dataclass, field
from omegaconf import DictConfig
from verl.utils.config import omega_conf_to_dataclass

@dataclass
class EngineConfig:
    """Dataclass for Engine configuration."""
    model: 'ModelConfig'
    optim: 'OptimConfig'
    checkpoint: 'CheckpointConfig'
    ppo_mini_batch_size: int
    ppo_micro_batch_size: int | None
    forward_micro_batch_size: int
    ppo_micro_batch_size_per_gpu: int | None
    forward_micro_batch_size_per_gpu: int | None
    ulysses_sequence_parallel_size: int = 1
    strategy: str = "fsdp"
    grad_clip: float | None = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int | None = None
    rollout_n: int = 1


@dataclass
class ModelConfig:
    """Dataclass for model configuration."""
    path: str
    tokenizer_path: str
    fsdp_config: 'SystemConfig'
    lora_rank: int = 0
    lora_alpha: int | None = None
    target_modules: list[str] | None = None
    trust_remote_code: bool = False
    custom_chat_template: str | None = None
    override_config: dict | None = None
    use_shm: bool = False
    enable_gradient_checkpointing: bool = False
    enable_activation_offload: bool = False
    use_remove_padding: bool = False
    external_lib: str | None = None


@dataclass
class OptimConfig:
    """Dataclass for optimizer configuration."""
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    total_training_steps: int = 0
    lr_warmup_steps: int = -1
    lr_warmup_steps_ratio: float = 0.0
    warmup_style: str = "constant"


@dataclass
class SystemConfig:
    """Dataclass for FSDP system configuration."""
    fsdp_size: int
    param_offload: bool
    optimizer_offload: bool
    wrap_policy: dict
    forward_prefetch: bool
    reshard_after_forward: bool
    model_dtype: str = "fp32"
    mixed_precision: dict | None = None
    offload_policy: bool = False

@dataclass
class CheckpointConfig:
    save_contents: list[str]
    load_contents: list[str]
    async_save: bool

def convert_critic_engine_config(hydra_config: DictConfig) -> EngineConfig:
    """
    Convert a Hydra config to the FSDPEngineConfig dataclass.

    Args:
        hydra_config (DictConfig): The Hydra config object.

    Returns:
        FSDPEngineConfig: The converted dataclass.
    """
    prev_config = omega_conf_to_dataclass(hydra_config)
    print(prev_config)
    model_config = ModelConfig(
        path=prev_config.model.path,
        tokenizer_path=prev_config.model.tokenizer_path,
        fsdp_config=SystemConfig(
            fsdp_size=prev_config.model.fsdp_config.fsdp_size,
            param_offload=prev_config.model.fsdp_config.param_offload,
            optimizer_offload=prev_config.model.fsdp_config.optimizer_offload,
            wrap_policy=prev_config.model.fsdp_config.wrap_policy,
            forward_prefetch=prev_config.model.fsdp_config.forward_prefetch,
            reshard_after_forward=prev_config.model.fsdp_config.reshard_after_forward,
            model_dtype=prev_config.model.fsdp_config.model_dtype,
            mixed_precision=prev_config.model.fsdp_config.get("mixed_precision", None),
            offload_policy=prev_config.model.fsdp_config.offload_policy
        ),
        lora_rank=prev_config.model.lora_rank,
        lora_alpha=prev_config.model.lora_alpha,
        target_modules=prev_config.model.target_modules,
        trust_remote_code=prev_config.model.trust_remote_code,
        custom_chat_template=prev_config.model.get("custom_chat_template", None),
        override_config=prev_config.model.override_config,
        use_shm=prev_config.model.use_shm,
        enable_gradient_checkpointing=prev_config.model.enable_gradient_checkpointing,
        enable_activation_offload=prev_config.model.enable_activation_offload,
        use_remove_padding=prev_config.model.use_remove_padding,
        external_lib=prev_config.model.external_lib
    )
    optim_config = OptimConfig(
        lr=prev_config.optim.lr,
        betas=prev_config.optim.get("betas", (0.9, 0.999)),
        weight_decay=prev_config.optim.weight_decay,
        total_training_steps=prev_config.optim.total_training_steps,
        lr_warmup_steps=prev_config.optim.lr_warmup_steps,
        lr_warmup_steps_ratio=prev_config.optim.lr_warmup_steps_ratio,
        warmup_style=prev_config.optim.get("warmup_style", "constant")
    )
    ret = EngineConfig(
        model=model_config,
        optim=optim_config,
        ppo_mini_batch_size=prev_config.ppo_mini_batch_size,
        ppo_micro_batch_size=prev_config.ppo_micro_batch_size,
        forward_micro_batch_size=prev_config.forward_micro_batch_size,
        ppo_micro_batch_size_per_gpu=prev_config.ppo_micro_batch_size_per_gpu,
        forward_micro_batch_size_per_gpu=prev_config.forward_micro_batch_size_per_gpu,
        ulysses_sequence_parallel_size=prev_config.ulysses_sequence_parallel_size,
        strategy=prev_config.strategy,
        grad_clip=prev_config.grad_clip,
        use_dynamic_bsz=prev_config.use_dynamic_bsz,
        ppo_max_token_len_per_gpu=prev_config.ppo_max_token_len_per_gpu,
        rollout_n=prev_config.rollout_n,
        checkpoint=prev_config.checkpoint
    )
    return ret
