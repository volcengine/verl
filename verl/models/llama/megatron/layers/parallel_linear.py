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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py

import torch
from megatron.core import tensor_parallel


class QKVParallelLinear(tensor_parallel.ColumnParallelLinear):
    def __init__(
        self,
        input_size,
        num_heads,
        num_key_value_heads,
        head_dim,
        *,
        bias=True,
        gather_output=True,
        skip_bias_add=False,
        **kwargs,
    ):
        # Keep input parameters, and already restrict the head numbers
        self.input_size = input_size
        self.q_output_size = num_heads * head_dim
        self.kv_output_size = num_key_value_heads * head_dim
        self.head_dim = head_dim
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add

        input_size = self.input_size
        output_size = (num_heads + 2 * num_key_value_heads) * self.head_dim

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            **kwargs,
        )


class MergedColumnParallelLinear(tensor_parallel.ColumnParallelLinear):
    def __init__(
        self,
        input_size,
        gate_ouput_size,
        up_output_size,
        *,
        bias=True,
        gather_output=True,
        skip_bias_add=False,
        **kwargs,
    ):
        # Keep input parameters, and already restrict the head numbers
        self.input_size = input_size
        self.output_size = gate_ouput_size + up_output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add

        super().__init__(
            input_size=self.input_size,
            output_size=self.output_size,
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            **kwargs,
        )


class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        bias=True,
    ):
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

    def forward(
        self,
        input_,
        weight=None,
        runtime_gather_output=None,
    ):
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None
