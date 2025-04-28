# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/config.py

import enum
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Union

from transformers import PretrainedConfig

# Add for verl
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.loader import BaseModelLoader

logger = init_logger(__name__)


class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    MEGATRON = "megatron"
    HF = "hf"
    DTENSOR = "dtensor"
    DUMMY_HF = "dummy_hf"
    DUMMY_MEGATRON = "dummy_megatron"
    DUMMY_DTENSOR = "dummy_dtensor"


class ModelConfig(ModelConfig):
    def __init__(self, hf_config: PretrainedConfig, *args, **kwargs) -> None:
        super().__init__(model=hf_config._name_or_path, tokenizer=hf_config._name_or_path, *args, **kwargs)  # noqa: B026
        self.hf_config = hf_config


@dataclass
class LoadConfig:
    """
    download_dir: Directory to download and load the weights, default to the
        default cache directory of huggingface.
    load_format: The format of the model weights to load:
        "auto" will try to load the weights in the safetensors format and
            fall back to the pytorch bin format if safetensors format is
            not available.
        "pt" will load the weights in the pytorch bin format.
        "safetensors" will load the weights in the safetensors format.
        "npcache" will load the weights in pytorch format and store
            a numpy cache to speed up the loading.
        "dummy" will initialize the weights with random values, which is
            mainly for profiling.
        "tensorizer" will use CoreWeave's tensorizer library for
            fast weight loading.
        "bitsandbytes" will load nf4 type weights.
    ignore_patterns: The list of patterns to ignore when loading the model.
        Default to "original/**/*" to avoid repeated loading of llama's
        checkpoints.

    """

    load_format: Union[str, LoadFormat, "BaseModelLoader"] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(default_factory=dict)
    ignore_patterns: Optional[Union[List[str], str]] = None

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = json.loads(model_loader_extra_config)
        self._verify_load_format()

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info("Ignoring the following patterns when downloading weights: %s", self.ignore_patterns)
        else:
            self.ignore_patterns = ["original/**/*"]

    def _verify_load_format(self) -> None:
        if not isinstance(self.load_format, str):
            return

        load_format = self.load_format.lower()
        self.load_format = LoadFormat(load_format)

        rocm_not_supported_load_format: List[str] = []
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [f for f in LoadFormat.__members__ if (f not in rocm_not_supported_load_format)]
            raise ValueError(f"load format '{load_format}' is not supported in ROCm. Supported load formats are {rocm_supported_load_format}")
