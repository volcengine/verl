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
from typing import List, Optional, Union
from dataclasses import dataclass, field, fields

from transformers import PretrainedConfig

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils import is_hip
# Add for verl
from vllm.config import ModelConfig, _get_and_verify_dtype, _get_and_verify_max_len

GPTQMarlinConfig = get_quantization_config("gptq_marlin")

logger = init_logger(__name__)

_GB = 1 << 30


class ModelConfig(ModelConfig):
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
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
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        code_revision: The specific revision to use for the model code on
            Hugging Face Hub. It can be a branch name, a tag name, or a
            commit id. If unspecified, will use the default version.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        quantization_param_path: Path to JSON file containing scaling factors.
            Used to load KV cache scaling factors into the model when KV cache
            type is FP8_E4M3 on ROCm (AMD GPU). In the future these will also
            be used to load activation and weight scaling factors when the
            model dtype is FP8_E4M3 on ROCm.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode (DEPRECATED. Use max_seq_len_to_capture instead).
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer.
        served_model_name: The model name used in metrics tag `model_name`,
            matches the model name exposed via the APIs. If multiple model 
            names provided, the first name will be used. If not specified, 
            the model name will be the same as `model`.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        dtype: str,
        seed: int,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        quantization_param_path: Optional[str] = None,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: Optional[int] = None,
        max_logprobs: int = 5,
        skip_tokenizer_init: bool = False,
        served_model_name: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self.model = hf_config._name_or_path
        self.tokenizer = hf_config._name_or_path
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.enforce_eager = enforce_eager
        self.max_context_len_to_capture = max_context_len_to_capture
        if self.max_context_len_to_capture is not None:
            raise ValueError("`max_context_len_to_capture` is deprecated. "
                             "Use `max_seq_len_to_capture` instead.")
        self.max_seq_len_to_capture = (max_seq_len_to_capture or max_context_len_to_capture)
        self.max_logprobs = max_logprobs
        self.skip_tokenizer_init = skip_tokenizer_init

        # self.hf_config = get_config(model, trust_remote_code, revision)
        self.hf_config = hf_config
        self.hf_text_config = get_hf_text_config(hf_config)
        # TODO: for multimodal model
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)
        self.max_model_len = _get_and_verify_max_len(self.hf_config, max_model_len)
        # self.served_model_name = get_served_model_name(model,
        #                                                served_model_name)
        # self._verify_load_format()
        # self._verify_tokenizer_mode()
        self._verify_quantization()
        self._verify_cuda_graph()


class LoadFormat(str, enum.Enum):
    AUTO = 'auto'
    MEGATRON = "megatron"
    HF = "hf"
    DTENSOR = 'dtensor'
    DUMMY_HF = 'dummy_hf'
    DUMMY_MEGATRON = 'dummy_megatron'
    DUMMY_DTENSOR = 'dummy_dtensor'


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
    """

    load_format: Union[str, LoadFormat, "BaseModelLoader"] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(default_factory=dict)

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = json.loads(model_loader_extra_config)
        self._verify_load_format()

    def _verify_load_format(self) -> None:
        if not isinstance(self.load_format, str):
            return

        load_format = self.load_format.lower()
        self.load_format = LoadFormat(load_format)

        rocm_not_supported_load_format: List[str] = []
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f for f in LoadFormat.__members__ if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(f"load format '{load_format}' is not supported in ROCm. "
                             f"Supported load formats are "
                             f"{rocm_supported_load_format}")
