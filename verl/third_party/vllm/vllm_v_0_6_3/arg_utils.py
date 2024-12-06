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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py

import os
from dataclasses import dataclass

from transformers import PretrainedConfig
from vllm.config import EngineConfig
from vllm.engine.arg_utils import EngineArgs

from .config import LoadConfig, ModelConfig


@dataclass
class EngineArgs(EngineArgs):
    model_hf_config: PretrainedConfig = None  # for verl

    def __post_init__(self):
        pass

    def create_model_config(self) -> ModelConfig:
        return ModelConfig(
            hf_config=self.model_hf_config,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            quantization_param_path=self.quantization_param_path,
            enforce_eager=self.enforce_eager,
            max_context_len_to_capture=self.max_context_len_to_capture,
            max_seq_len_to_capture=self.max_seq_len_to_capture,
            max_logprobs=self.max_logprobs,
            disable_sliding_window=self.disable_sliding_window,
            skip_tokenizer_init=self.skip_tokenizer_init,
            served_model_name=self.served_model_name,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            use_async_output_proc=not self.disable_async_output_proc,
            override_neuron_config=self.override_neuron_config,
            config_format=self.config_format,
            mm_processor_kwargs=self.mm_processor_kwargs,
        )

    def create_load_config(self) -> LoadConfig:
        return LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
        )

    def create_engine_config(self) -> EngineConfig:
        engine_config = super().create_engine_config()

        # NOTE[VERL]: Use the world_size set by torchrun
        world_size = int(os.getenv("WORLD_SIZE", "-1"))
        assert world_size != -1, "The world_size is set to -1, not initialized by TORCHRUN"
        engine_config.parallel_config.world_size = world_size

        return engine_config
