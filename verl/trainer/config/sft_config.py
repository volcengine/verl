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

from hydra.core.config_store import ConfigStore

from verl.base_config import BaseConfig
from verl.trainer.config.config import CheckpointConfig
from verl.workers.config.engine import AllEngineConfig
from verl.workers.config.model import HFModelConfig
from verl.workers.config.optimizer import AllOptimizerConfig


@dataclass
class CustomClassConfig(BaseConfig):
    """Configuration for custom dataset class."""

    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class SFTDataConfig(BaseConfig):
    """Configuration for SFT training data."""

    # Batch size configurations
    train_batch_size: int = 256  # global batch size
    micro_batch_size_per_gpu: int = 4  # this is also val batch size
    max_token_len_per_gpu: int = 8192
    use_dynamic_bsz: bool = True

    # Data file paths
    train_files: list[str] = field(default_factory=lambda: ["~/data/gsm8k/train.parquet"])
    val_files: list[str] = field(default_factory=lambda: ["~/data/gsm8k/test.parquet"])

    # Multi-turn settings
    messages_key: str = "messages"  # Key for messages list in multi-turn mode
    tools_key: str = "tools"  # Key for tools list in multi-turn mode
    enable_thinking_key: str = "enable_thinking"  # Whether to enable thinking in multi-turn mode

    # Padding configurations
    pad_mode: str = "left_right"
    max_length: int = 1024  # for right padding
    max_prompt_length: int = 512  # for left right padding
    max_response_length: int = 512  # for left right padding
    truncation: str = "error"

    # Other configurations
    balance_dp_token: bool = False  # to be implement
    use_shm: bool = False
    apply_chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    # Custom dataset class
    custom_cls: CustomClassConfig = field(default_factory=CustomClassConfig)


@dataclass
class SFTTrainerConfig(BaseConfig):
    """Configuration for SFT trainer."""

    # Directory configurations
    default_local_dir: str = "checkpoints/${trainer.project_name}/${trainer.experiment_name}"
    default_hdfs_dir: Optional[str] = None

    # Project and experiment naming
    project_name: str = "gsm8k-sft"
    experiment_name: str = "test"

    # Training configurations
    total_epochs: int = 4
    total_training_steps: Optional[int] = None
    seed: int = 1

    # Logging and monitoring
    logger: list[str] = field(default_factory=lambda: ["console", "wandb"])

    # Checkpoint configurations
    save_freq: int | str = -1
    test_freq: int | str = -1
    max_ckpt_to_keep: Optional[int] = None

    # Resume configurations
    resume_mode: str = "auto"  # "auto", "disable", or "resume_path"
    resume_from_path: Optional[str] = None

    # Device configuration
    device: str = "cuda"


@dataclass
class SFTConfig(BaseConfig):
    """Main SFT training configuration."""

    # Sub-configurations
    data: SFTDataConfig = field(default_factory=SFTDataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    trainer: SFTTrainerConfig = field(default_factory=SFTTrainerConfig)
    model: HFModelConfig = field(default_factory=HFModelConfig)
    engine: AllEngineConfig = field(default_factory=AllEngineConfig)
    optim: AllOptimizerConfig = field(default_factory=AllOptimizerConfig)

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate configuration
        if self.trainer.total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        if self.data.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")


# Note: ConfigStore registration moved to register_sft_configs() function
# to avoid loading model during import


# Create some preset configurations
@dataclass
class SFTDebugConfig(SFTConfig):
    """Debug configuration with smaller settings."""

    def __post_init__(self):
        super().__post_init__()
        # Override for debugging
        self.data.train_batch_size = 4
        self.data.micro_batch_size_per_gpu = 1
        self.trainer.total_epochs = 1
        self.trainer.save_freq = 10


@dataclass
class SFTLargeModelConfig(SFTConfig):
    """Configuration for large model training."""

    def __post_init__(self):
        super().__post_init__()
        # Override for large models
        self.data.micro_batch_size_per_gpu = 1
        self.data.max_token_len_per_gpu = 4096
        self.engine.param_offload = True
        self.engine.optimizer_offload = True


def register_sft_configs():
    """Register SFT configurations with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="sft_config", node=SFTConfig)
