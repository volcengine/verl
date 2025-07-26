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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import BaseModelConfig
from verl.utils.profiler import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig

__all__ = ["RewardModelConfig", "FSDPRewardModelConfig", "FSDPRewardModelCfg", "McoreRewardModelConfig"]


@dataclass
class RewardModelConfig(BaseConfig):
    """Configuration for reward model inference.
    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.
    Args:
        strategy (str): Strategy used for reward model inference (fsdp, fsdp2, megatron).
        model (Dict[str, Any]): Model configuration for reward scoring, including path, tokenizer_path, etc.
        micro_batch_size (Optional[int]): Global micro batch size (deprecated).
        micro_batch_size_per_gpu (Optional[int]): Local per-GPU micro batch size.
        max_length (Optional[int]): Maximum sequence length to process for scoring.
        use_dynamic_bsz (bool): Whether to automatically adjust batch size at runtime.
        forward_max_token_len_per_gpu (int): Max tokens per GPU in one forward pass.
        reward_manager (str): Reward Manager. This defines the mechanism of computing rule-based reward and
            handling different reward sources.
        launch_reward_fn_async (bool): Whether to launch custom reward function asynchronously during log_prob.
        sandbox_fusion (Dict[str, Any]): Cloud/local sandbox fusion configuration for custom reward logic.
        profiler (Dict[str, Any]): Profiler configuration.
        enable (Optional[bool]): Whether to enable the reward model.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "micro_batch_size_per_gpu",
        "micro_batch_size",
    }

    strategy: str = MISSING
    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    micro_batch_size_per_gpu: Optional[int] = None
    micro_batch_size: Optional[int] = None
    max_length: Optional[int] = None
    use_dynamic_bsz: bool = False
    forward_max_token_len_per_gpu: int = 32768
    reward_manager: str = "naive"
    launch_reward_fn_async: bool = False
    sandbox_fusion: dict[str, Any] = field(default_factory=dict)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    enable: Optional[bool] = None

    def __post_init__(self):
        """Validate reward model configuration parameters."""
        assert self.strategy != MISSING
        if not self.use_dynamic_bsz:
            self._check_mutually_exclusive(self.micro_batch_size, self.micro_batch_size_per_gpu, "reward_model")

    def validate(self, n_gpus: int):
        """Validate reward model configuration with runtime parameters.
        Args:
            n_gpus: Total number of GPUs available
        """
        pass

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.
        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.
        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.
        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        param = "micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreRewardModelConfig(RewardModelConfig):
    """Configuration for Megatron-based reward model inference.

    The inheritance from RewardModelConfig provides all base reward model configuration plus Megatron-specific settings.

    Args:
        nccl_timeout (int): NCCL timeout in seconds for distributed operations.
        megatron (Dict[str, Any]): Megatron-specific parallelism settings.
        load_weight (bool): Whether to load initial weights.
    """

    strategy: str = "megatron"
    nccl_timeout: int = 600
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    load_weight: bool = True

    def validate(self, n_gpus: int):
        """Validate Megatron reward model configuration with runtime parameters."""
        super().validate(n_gpus)


@dataclass
class FSDPRewardModelConfig(RewardModelConfig):
    """Configuration for FSDP-based reward model inference.
    The inheritance from RewardModelConfig provides all base reward model configuration plus FSDP-specific settings.
    Args:
        ulysses_sequence_parallel_size (int): Sequence parallelism size for Ulysses-style model parallelism.
    """

    strategy: str = "fsdp"
    ulysses_sequence_parallel_size: int = 1

    def __post_init__(self):
        """Validate FSDP reward model configuration parameters."""
        super().__post_init__()

        if self.strategy in {"fsdp", "fsdp2"}:
            if self.ulysses_sequence_parallel_size > 1:
                if not self.model.get("use_remove_padding", False):
                    raise ValueError(
                        "When using sequence parallelism for reward model, you must enable `use_remove_padding`."
                    )

    def validate(self, n_gpus: int):
        """Validate FSDP reward model configuration with runtime parameters."""
        super().validate(n_gpus)

        if not self.use_dynamic_bsz:
            sp_size = self.ulysses_sequence_parallel_size
            if self.micro_batch_size is not None:
                if self.micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"reward_model.micro_batch_size ({self.micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )


@dataclass
class FSDPRewardModelCfg(BaseModelConfig):
    """FSDP-enabled reward model configuration.
    Inherits base reward model settings and adds distributed-memory.
    Args:
        input_tokenizer (Optional[str]): Input tokenizer. If the reward model's chat template is inconsistent with
            the policy, we need to first decode to plaintext, then apply the rm's chat_template. Then score with RM.
            If chat_template are consistent, it can be set to null. set this to null if the chat template is identical.
        use_shm (bool): Whether to use shared memory for loading the model.
        use_remove_padding (bool): Use remove-padding optimization (saves compute).
        use_fused_kernels (bool): Whether to use fused reward kernels for speedup.
        fsdp_config (FSDPEngineConfig): FSDP-specific configuration block.
    """

    input_tokenizer: Optional[str] = None
    use_shm: bool = False
    use_remove_padding: bool = False
    use_fused_kernels: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
