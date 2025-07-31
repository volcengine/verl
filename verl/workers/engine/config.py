from dataclasses import dataclass, field
from omegaconf import DictConfig
from verl.utils.config import omega_conf_to_dataclass

@dataclass
class EngineConfig:
    """Dataclass for Engine configuration."""
    model: 'ModelConfig'
    optim: 'OptimConfig'
    system: 'SystemConfig'
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
    forward_max_token_len_per_gpu: int
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
        external_lib=config.external_lib
    )
    return model_config


def get_optim_config(config):
    optim_config = OptimConfig(
        lr=config.lr,
        betas=config.get("betas", (0.9, 0.999)),
        weight_decay=config.weight_decay,
        total_training_steps=config.total_training_steps,
        lr_warmup_steps=config.lr_warmup_steps,
        lr_warmup_steps_ratio=config.lr_warmup_steps_ratio,
        warmup_style=config.get("warmup_style", "constant")
    )
    return optim_config


def get_system_config(config):
    system_config = SystemConfig(
        fsdp_size=config.fsdp_size,
        param_offload=config.param_offload,
        optimizer_offload=config.optimizer_offload,
        wrap_policy=config.wrap_policy,
        forward_prefetch=config.forward_prefetch,
        reshard_after_forward=config.reshard_after_forward,
        model_dtype=config.get("model_dtype", "f32"),
        mixed_precision=config.get("mixed_precision", None),
        offload_policy=config.offload_policy
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
                      rollout_n):
    engine_config = EngineConfig(
        model=model_config,
        optim=optim_config,
        system=system_config,
        checkpoint=checkpoint_config,
        ppo_mini_batch_size=config.ppo_mini_batch_size,
        ppo_micro_batch_size=config.ppo_micro_batch_size,
        forward_micro_batch_size=config.get("forward_micro_batch_size", None),                  # no such config for actor
        ppo_micro_batch_size_per_gpu=config.ppo_micro_batch_size_per_gpu,
        forward_micro_batch_size_per_gpu=config.get("forward_micro_batch_size_per_gpu", None),  # no such config for actor
        ulysses_sequence_parallel_size=config.ulysses_sequence_parallel_size,
        strategy=config.strategy,
        grad_clip=config.grad_clip,
        use_dynamic_bsz=config.use_dynamic_bsz,
        ppo_max_token_len_per_gpu=config.ppo_max_token_len_per_gpu,
        forward_max_token_len_per_gpu=config.get("forward_max_token_len_per_gpu", None),        # no such config for actor
        rollout_n=rollout_n
    )
    return engine_config



def engine_config_for_actor(actor_config: DictConfig) -> EngineConfig:
    """
    Convert a Hydra config to the FSDPEngineConfig dataclass.

    Args:
        actor_config (DictConfig): The config from CriticWorker.

    Returns:
        FSDPEngineConfig: The converted dataclass.
    """
    print(actor_config)
    model_config = get_model_config(actor_config.model)
    optim_config = get_optim_config(actor_config.actor.optim)
    system_config = get_system_config(actor_config.actor.fsdp_config)
    ckpt_config = get_checkpoint_config(actor_config.actor.checkpoint)

    ret = get_engine_config(actor_config.actor,
                            model_config,
                            optim_config,
                            system_config,
                            ckpt_config)
    return ret