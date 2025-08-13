from dataclasses import dataclass, fields, MISSING
from omegaconf import DictConfig
from verl.utils.config import omega_conf_to_dataclass
from typing import Any
import copy

"""
Basic idea:
- unify the config feed into the engine by different workers like Actor/Critic
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
    strategy: str
    model: 'ModelConfig'
    optim: 'OptimConfig'
    system: 'SystemConfig'
    checkpoint: 'CheckpointConfig'
    total_training_steps: int
    use_dynamic_bsz: bool
    train_mini_batch_size: int
    train_micro_batch_size_per_gpu: int
    train_max_token_len_per_gpu: int
    infer_micro_batch_size_per_gpu: int
    infer_max_token_len_per_gpu: int


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    """Dataclass for model configuration."""
    path: str
    module_type: str
    lora_rank: int
    lora_alpha: int
    target_modules: list[str]
    trust_remote_code: bool
    use_shm: bool
    external_lib: str
    override_config: dict       = None
    custom_chat_template: str   = None
    tokenizer_path: str         = None
    use_liger: bool             = False
    use_fused_kernels: bool     = False
    fused_kernel_options: dict  = None


@dataclass(frozen=True)
class OptimConfig(BaseConfig):
    """Dataclass for optimizer configuration."""
    grad_clip: float
    betas: tuple[float, float]                      # for AdamW optimizer
    weight_decay: float                             # for AdamW optimizer
    lr: float
    lr_warmup_steps : int           = 0
    lr_warmup_steps_ratio: float    = 0.0
    lr_scheduler_style: str         = "constant"
    lr_scheduler_args: dict         = None
    # min_lr_ratio: float                           # for cosine scheduler
    # num_cycles: float                             # for cosine scheduler


# TODO: use inheritance for different backend
# - FSDPSystemConfig(SystemConfig)
# - MCoreSystemConfig(SystemConfig)

@dataclass(frozen=True)
class SystemConfig(BaseConfig):
    """Dataclass for FSDP system configuration."""
    fsdp_size: int                      = 1
    model_dtype: str                    = "fp32"
    param_offload: bool                 = False
    optimizer_offload: bool             = False
    wrap_policy: dict                   = None
    reshard_after_forward: bool         = False
    offload_policy: bool                = False
    forward_prefetch: bool              = False
    ulysses_sequence_parallel_size: int = 1
    enable_gradient_checkpointing: bool = False
    enable_activation_offload: bool     = False
    use_remove_padding: bool            = False
    mixed_precision: dict               = None
    use_orig_params: bool               = False


@dataclass(frozen=True)
class CheckpointConfig(BaseConfig):
    save_contents: list[str]
    load_contents: list[str]
    async_save: bool                    = False


def generate_config(new_config_cls, prev_config, provided_fields):
    field_items = {}
    for name, f in new_config_cls.__dataclass_fields__.items():
        field_items[name] = f

    required_fields = field_items.keys()
    kwargs = copy.deepcopy(provided_fields)
    missing_fields = required_fields - set(kwargs.keys())

    for field in missing_fields:
        if field in prev_config:
            kwargs[field] = prev_config.get(field)
        elif field_items[field].default != MISSING:
            default_value = field_items[field].default
            print(f"{new_config_cls.__name__} missing field: " \
                  f"`{field}`, will try to apply default value {default_value}.")
        else:
            raise ValueError(f"{new_config_cls.__name__} missing field: " \
                             f"`{field}`.")
    return new_config_cls(**kwargs)

def get_model_config(config, **kwargs):
    return generate_config(ModelConfig, config, kwargs)


def get_optim_config(config, **kwargs):
    return generate_config(OptimConfig, config, kwargs)


def get_system_config(config, **kwargs):
    return generate_config(SystemConfig, config, kwargs)


def get_checkpoint_config(config, **kwargs):
    return generate_config(CheckpointConfig, config, kwargs)


def get_engine_config(config,
                      model_config,
                      optim_config,
                      system_config,
                      checkpoint_config,
                      total_training_steps,
                      use_dynamic_bsz,
                      train_mini_batch_size,
                      train_micro_batch_size_per_gpu,
                      train_max_token_len_per_gpu,
                      infer_micro_batch_size_per_gpu,
                      infer_max_token_len_per_gpu):
    engine_config = EngineConfig(
        strategy=config.strategy,
        model=model_config,
        optim=optim_config,
        system=system_config,
        checkpoint=checkpoint_config,
        total_training_steps=total_training_steps,
        use_dynamic_bsz=use_dynamic_bsz,
        train_mini_batch_size=train_mini_batch_size,
        train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu,
        train_max_token_len_per_gpu=train_max_token_len_per_gpu,
        infer_micro_batch_size_per_gpu=infer_micro_batch_size_per_gpu,
        infer_max_token_len_per_gpu=infer_max_token_len_per_gpu
    )
    return engine_config