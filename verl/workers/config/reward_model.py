# verl/workers/config/reward_model.py

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig
from .actor import ActorConfig

__all__ = [
    "SandboxFusionConfig",
    "RewardModelRolloutEngineKwargsVLLM",
    "RewardModelRolloutEngineKwargsSGLang",
    "RewardModelRolloutEngineKwargs",
    "RewardModelRolloutValKwargs",
    "RewardModelRolloutTraceConfig",
    "RewardModelRolloutConfig",
    "RewardModelDataProcessorConfig",
    "RewardModelFSDPConfig",
    "RewardModelInnerConfig",
    "RewardModelConfig",
]


@dataclass
class SandboxFusionConfig(BaseConfig):
    """Configuration for sandbox fusion for code execution rewards."""
    url: Optional[str] = None
    max_concurrent: int = 64
    memory_limit_mb: int = 1024

@dataclass
class RewardModelRolloutEngineKwargsVLLM(BaseConfig):
    kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "swap_space": None,
        "disable_mm_preprocessor_cache": False
    })

@dataclass
class RewardModelRolloutEngineKwargsSGLang(BaseConfig):
    kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "attention_backend": None
    })
@dataclass
class RewardModelRolloutEngineKwargs(BaseConfig):
    vllm: Dict[str, Any] = field(default_factory=lambda: {
        "swap_space": None,
        "disable_mm_preprocessor_cache": False
    })
    sglang: Dict[str, Any] = field(default_factory=lambda: {
        "attention_backend": None
    })
    
@dataclass
class RewardModelRolloutValKwargs(BaseConfig):
    top_k: int = -1
    top_p: float = 1.0
    temperature: float = 0.0
    n: int = 1
    do_sample: bool = False

@dataclass
class RewardModelRolloutTraceConfig(BaseConfig):
    backend: Optional[str] = None
    token2text: bool = False

@dataclass
class RewardModelRolloutConfig(BaseConfig):
    """Configuration for the rollout engine within the Reward Model."""
    name: str = "vllm"
    mode: str = "sync"
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    prompt_length: int = 512
    response_length: int = 512
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    free_cache_engine: bool = True
    load_format: str = "dummy_dtensor"
    layered_summon: bool = False
    tensor_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384
    disable_log_stats: bool = True
    enable_chunked_prefill: bool = True
    do_sample: bool = False
    n: int = 1
    multi_stage_wake_up: bool = False
    calculate_log_probs: bool = False
    engine_kwargs: RewardModelRolloutEngineKwargs = field(default_factory=RewardModelRolloutEngineKwargs)
    val_kwargs: RewardModelRolloutValKwargs = field(default_factory=RewardModelRolloutValKwargs)
    trace: RewardModelRolloutTraceConfig = field(default_factory=RewardModelRolloutTraceConfig)

@dataclass
class RewardModelDataProcessorConfig(BaseConfig):
    """Configuration for data processing functions."""
    path: Optional[str] = "verl/utils/reward_process.py"
    preprocess_fn_name: Optional[str] = "reward_preprocess"
    postprocess_fn_name: Optional[str] = "reward_postprocess"

@dataclass
class RewardModelFSDPConfig(BaseConfig):
    """FSDP specific settings for the Reward Model."""
    min_num_params: int = 0
    param_offload: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False
    wrap_policy: Dict[str, Any] = field(default_factory=lambda: {
        "min_num_params": 0,
    })

@dataclass
class RewardModelInnerConfig(BaseConfig):
    """Configuration for the inner model component of the Reward Model."""
    path: str = MISSING
    input_tokenizer: Optional[str] = None
    use_shm: bool = False
    external_lib: Optional[str] = None
    use_remove_padding: bool = False
    use_fused_kernels: bool = False
    trust_remote_code: bool = False
    data_processer: RewardModelDataProcessorConfig = field(default_factory=RewardModelDataProcessorConfig)
    fsdp_config: RewardModelFSDPConfig = field(default_factory=RewardModelFSDPConfig)
    rollout: RewardModelRolloutConfig = field(default_factory=RewardModelRolloutConfig)

@dataclass
class RewardModelConfig(BaseConfig):
    """Configuration for the Reward Model worker."""
    enable: bool = False
    rm_mode: str = "discriminator"
    strategy: str = "fsdp"
    micro_batch_size: Optional[int] = None
    micro_batch_size_per_gpu: Optional[int] = None
    max_length: Optional[int] = None
    ulysses_sequence_parallel_size: int = 1
    use_dynamic_bsz: bool = False
    forward_max_token_len_per_gpu: int = 16384
    reward_manager: str = "naive"
    launch_reward_fn_async: bool = False
    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
    model: RewardModelInnerConfig = field(default_factory=RewardModelInnerConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    
    def __post_init__(self):            
        
        if not self.enable:
            return

        """Validates the configuration after initialization."""
        if self.rm_mode not in ("generator", "discriminator"):
            raise ValueError(f"Config Error: Invalid rm_mode '{self.rm_mode}'.")

        if not self.use_dynamic_bsz:
            ActorConfig._check_mutually_exclusive(
                self.micro_batch_size, self.micro_batch_size_per_gpu, name="reward_model"
            )

