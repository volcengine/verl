# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import os

import numpy as np
import pytest

from verl.protocol import DataProto
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_omni_rollout_spmd import vLLMOmniRollout


@pytest.fixture
def mock_data() -> DataProto:
    test_prompt = "a photo of a cat"
    test_prompt_2 = "a photo of a dog"
    data = DataProto(non_tensor_batch={"prompt": np.array([test_prompt, test_prompt_2])})
    return data


class TestvLLMOmniRollout:
    def setup_class(self):
        model_path = os.path.expanduser("~/models/Qwen/Qwen-Image")
        tokenizer_path = os.path.join(model_path, "tokenizer")

        diffusion_config = RolloutConfig()
        model_config = HFModelConfig(path=model_path, tokenizer_path=tokenizer_path)
        self.rollout_engine = vLLMOmniRollout(diffusion_config, model_config, None)

    def test_generate_sequences(self, mock_data: DataProto):
        result = self.rollout_engine.generate_sequences(mock_data)
        expected_batch_keys = [
            "responses",
            "latents",
            "timesteps",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
        for key in expected_batch_keys:
            assert key in result.batch, f"Key {key} not found in result batch."

        assert result.batch.batch_size[0] == 1, f"Expected batch size 1, got {result.batch.batch_size[0]}."
        assert "cached_steps" in result.meta_info, "cached_steps not found in meta_info."
