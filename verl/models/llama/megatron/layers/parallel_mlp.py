# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core import ModelParallelConfig
from torch import nn
from transformers.activations import ACT2FN
from verl.models.llama.megatron.layers.parallel_linear import MergedColumnParallelLinear

from verl.utils.megatron import tensor_parallel as tp_utils


class ParallelLlamaMLP(nn.Module):

    def __init__(self, config, megatron_config: ModelParallelConfig = None) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # The weight is only [hidden_size, intermediate_size // model_parallel_world_size]

        column_kwargs = tp_utils.get_default_kwargs_for_column_parallel_linear()
        row_kwargs = tp_utils.get_default_kwargs_for_row_parallel_linear()

        if megatron_config is not None:
            assert column_kwargs.get('config', False), 'must have ModelParallelConfig'
            assert row_kwargs.get('config', False), 'must have ModelParallelConfig'
            tp_utils.update_kwargs_with_config(row_kwargs, megatron_config)
            tp_utils.update_kwargs_with_config(column_kwargs, megatron_config)

        tp_size = mpu.get_tensor_model_parallel_world_size()

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            gate_ouput_size=self.intermediate_size,
            up_output_size=self.intermediate_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            **column_kwargs,
        )
        self.gate_size = self.intermediate_size // tp_size

        self.down_proj = tensor_parallel.RowParallelLinear(input_size=self.intermediate_size,
                                                           output_size=self.hidden_size,
                                                           bias=False,
                                                           input_is_parallel=True,
                                                           skip_bias_add=False,
                                                           **row_kwargs)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate_up = self.gate_up_proj(x)[0]
        gate, up = gate_up.split(self.gate_size, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)[0]
