#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import pytest

from verl.workers.config import DeepSpeedEngineConfig, DeepSpeedOptimizerConfig


def test_invalid_ulysses():
    with pytest.raises(ValueError):
        DeepSpeedEngineConfig(ulysses_sequence_parallel_size=0)


def test_invalid_model_dtype():
    with pytest.raises(ValueError):
        DeepSpeedEngineConfig(model_dtype="int8")


def test_invalid_mixed_precision_type():
    with pytest.raises(ValueError):
        DeepSpeedEngineConfig(mixed_precision=123)  # Must be dict, str, or None


def test_valid_mixed_precision_types():
    # These should not raise errors
    DeepSpeedEngineConfig(mixed_precision=None)
    DeepSpeedEngineConfig(mixed_precision={"param_dtype": "bf16"})
    DeepSpeedEngineConfig(mixed_precision="bf16")
    DeepSpeedEngineConfig(mixed_precision="fp16")


def test_optimizer_lr():
    with pytest.raises(ValueError):
        DeepSpeedOptimizerConfig(lr=0)


def test_optimizer_betas():
    with pytest.raises(ValueError):
        DeepSpeedOptimizerConfig(betas=(1.1, 0.9))


def test_optimizer_weight_decay():
    with pytest.raises(ValueError):
        DeepSpeedOptimizerConfig(weight_decay=-0.1)
