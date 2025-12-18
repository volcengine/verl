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

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig

from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "FSDPEngineConfig",
    "McoreEngineConfig",
    "DeepSpeedEngineConfig",
    "DeepSpeedOptimizerConfig",
    "TrainingWorkerConfig",
]


@dataclass
class EngineConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {
        "use_dynamic_bsz",
        "max_token_len_per_gpu",
        "micro_batch_size_per_gpu",
        "infer_max_token_len_per_gpu",
        "infer_micro_batch_size_per_gpu",
        "use_fused_kernels",
        "use_remove_padding",
    }

    # whether to offload param
    param_offload: bool = False
    # whether to offload optimizer
    optimizer_offload: bool = False
    # whether to offload grad
    grad_offload: bool = False
    # whether the engine is forward only (e.g., ref policy)
    forward_only: bool = False
    # the strategy (backend)
    strategy: str = None
    # model dtype
    dtype: str = "bfloat16"  # ["bfloat16", "float16"]
    # whether to use dynamic bsz
    use_dynamic_bsz: bool = True
    # for training
    max_token_len_per_gpu: int = None
    micro_batch_size_per_gpu: int = None
    # for inference
    infer_max_token_len_per_gpu: int = None
    infer_micro_batch_size_per_gpu: int = None
    # whether use fuse lm head kernel
    use_fused_kernels: bool = False
    # TODO (this may conflict with the one in model config)
    use_remove_padding: bool = True

    seed: int = 42

    full_determinism: bool = False

    def __post_init__(self):
        pass
        # TODO: turn on this check after we reorg config
        # if self.use_dynamic_bsz:
        #     assert self.max_token_len_per_gpu is not None
        # else:
        #     assert self.micro_batch_size_per_gpu is not None


@dataclass
class McoreEngineConfig(EngineConfig):
    """Configuration for Megatron parallelism.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        param_offload (bool): Whether to offload parameters to CPU.
        grad_offload (bool): Whether to offload gradients to CPU.
        optimizer_offload (bool): Whether to offload optimizer states to CPU.
        tensor_model_parallel_size (int): Tensor model parallel size.
        expert_model_parallel_size (int): Expert model parallel size for MoE models.
        expert_tensor_parallel_size (Optional[int]): Expert tensor parallel size for MoE models.
        pipeline_model_parallel_size (int): Pipeline model parallel size.
        virtual_pipeline_model_parallel_size (Optional[int]): Virtual pipeline model parallel size
            for interleaved scheduling.
        context_parallel_size (int): Context parallel size for long sequences.
        sequence_parallel (bool): Whether to enable sequence parallelism.
        use_distributed_optimizer (bool): Whether to use distributed optimizer.
        use_dist_checkpointing (bool): Whether to use distributed checkpointing.
        dist_checkpointing_path (Optional[str]): Path for distributed checkpointing.
        seed (int): Random seed for reproducibility.
        override_ddp_config (dict[str, Any]): Override configuration for DDP.
        override_transformer_config (dict[str, Any]): Override configuration for transformer.
        use_mbridge (bool): Whether to use MBridge for communication.
        dtype (str): Mixed precision training param dtype, default "bfloat16"
    """

    # sequence_parallel is not listed as a frozen field for auto-correction purpose
    _mutable_fields = EngineConfig._mutable_fields | {"sequence_parallel"}
    # mcore parallelism
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True
    use_dist_checkpointing: bool = False
    dist_checkpointing_path: Optional[str] = None
    dist_checkpointing_prefix: str = ""
    override_ddp_config: dict[str, Any] = field(default_factory=dict)
    override_transformer_config: dict[str, Any] = field(default_factory=dict)
    override_mcore_model_config: dict[str, Any] = field(default_factory=dict)
    use_mbridge: bool = True
    vanilla_mbridge: bool = True
    strategy: str = "megatron"

    def __post_init__(self) -> None:
        super().__post_init__()
        """config validation logics go here"""
        assert self.strategy == "megatron"
        assert self.dtype in ["bfloat16", "float16"], f"dtype {self.dtype} not supported"
        if self.tensor_model_parallel_size == 1:
            warnings.warn("set sequence parallel to false as TP size is 1", stacklevel=2)
            self.sequence_parallel = False


@dataclass
class FSDPEngineConfig(EngineConfig):
    """Configuration for FSDP (Fully Sharded Data Parallel).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        wrap_policy (Dict[str, Any]): Configuration for FSDP wrap policy.
        param_offload (bool): Whether to offload parameters to CPU, default False
        optimizer_offload (bool): Whether to offload optimizer states to CPU, default False
        offload_policy (bool): Whether to offload policy model parameters, default False
        reshard_after_forward (bool): Whether to reshard parameters after forward pass, default True
        fsdp_size (int): FSDP group size. -1 means use all available GPUs.
        forward_prefetch (bool): Whether to prefetch parameters for next forward pass, default False
        model_dtype (str): Model data type used to initialize the transformers model. default "fp32"
        use_orig_params (bool): Whether to use original parameters when initialize FSDP1, default False
        seed (int): Random seed for reproducibility.
        full_determinism (bool): If true, enable_full_determinism is called to ensure reproducible results
            in distributed training. Important: this will negatively impact performance, so only use it for
            debugging.
        mixed_precision (Optional[dict[str, Any]]): Mixed precision configuration for FSDP, default None
        dtype (str): Mixed precision training param dtype, default "bfloat16"
    """

    # ulysses_sequence_parallel_size is mutable for backward compatibility
    _mutable_fields = EngineConfig._mutable_fields | {"ulysses_sequence_parallel_size"}

    # fsdp specific flags
    wrap_policy: dict[str, Any] = field(default_factory=dict)
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False
    model_dtype: str = "fp32"
    use_orig_params: bool = False
    mixed_precision: Optional[dict[str, Any]] = None
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False
    strategy: str = "fsdp"

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy in ["fsdp", "fsdp2"], f"strategy {self.strategy} not supported"


# ---------------- DeepSpeed Configs -----------------
@dataclass
class DeepSpeedEngineConfig(BaseConfig):
    """Configuration for DeepSpeed engine (minimal subset)."""

    # offload & parallel
    param_offload: bool = False
    optimizer_offload: bool = False
    ulysses_sequence_parallel_size: int = 1

    # dtype / precision controls
    model_dtype: str = "fp32"  # initial parameter dtype
    mixed_precision: Optional[dict[str, Any]] = None  # e.g. {"param_dtype": "bf16"}
    forward_only: bool = False

    # features
    use_torch_compile: bool = False
    entropy_from_logits_with_chunking: bool = False

    # placeholder for parity with FSDP config fields accessed in engine_impl
    strategy: str = field(default="deepspeed", init=False)

    def __post_init__(self):
        # basic field validation
        if self.ulysses_sequence_parallel_size < 1:
            raise ValueError("ulysses_sequence_parallel_size must be >= 1")
        if self.model_dtype not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"Unsupported model_dtype {self.model_dtype}")
        # mixed_precision can be: None | str ("fp16"/"bf16") | dict
        if self.mixed_precision is not None and not isinstance(self.mixed_precision, dict | str):
            raise ValueError("mixed_precision must be a dict, str, or None")


@dataclass
class DeepSpeedOptimizerConfig(BaseConfig):
    """Optimizer config for DeepSpeed wrapper."""

    optimizer: str = "AdamW"
    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    eps: float = 1e-8
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    lr_warmup_steps: int = -1

    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0 < self.betas[0] < 1 and 0 < self.betas[1] < 1):
            raise ValueError("betas must each be in (0,1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")


@dataclass
class TrainingWorkerConfig(BaseConfig):
    model_type: str = None  # model type (language_model/value_model)
    model_config: HFModelConfig = None
    engine_config: EngineConfig = None
    optimizer_config: OptimizerConfig = None
    checkpoint_config: CheckpointConfig = None
