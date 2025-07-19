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

from verl.workers.config.optimizer import FSDPOptimizerConfig


class TestFSDPOptimizerConfigCPU:
    def test_default_configuration(self):
        config = FSDPOptimizerConfig()
        assert config.min_lr_ratio is None
        assert config.warmup_style == "constant"
        assert config.num_cycles == 0.5

    @pytest.mark.parametrize("warmup_style", ["constant", "cosine"])
    def test_valid_warmup_styles(self, warmup_style):
        config = FSDPOptimizerConfig(warmup_style=warmup_style)
        assert config.warmup_style == warmup_style

    def test_invalid_warmup_style(self):
        with pytest.raises(ValueError):
            FSDPOptimizerConfig(warmup_style="invalid_style")

    @pytest.mark.parametrize("num_cycles", [0.1, 1.0, 2.5])
    def test_num_cycles_configuration(self, num_cycles):
        config = FSDPOptimizerConfig(num_cycles=num_cycles)
        assert config.num_cycles == num_cycles

    def test_min_lr_ratio_validation(self):
        config = FSDPOptimizerConfig(min_lr_ratio=0.1)
        assert config.min_lr_ratio == 0.1

        with pytest.raises(ValueError):
            FSDPOptimizerConfig(min_lr_ratio=-0.1)
