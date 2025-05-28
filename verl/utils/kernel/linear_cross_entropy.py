#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

import typing

import torch
import torch.distributed as dist

from . import kernels


class LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, reduction: typing.Optional[str] = "mean", temperature: typing.Optional[float] = 1.0, dist_process_group: typing.Optional[dist.ProcessGroup] = None) -> typing.List[torch.Tensor]:
        """_summary_

        Args:
            ctx (_type_): _description_
            hidden (torch.Tensor): (batch_size, num_tokens, hidden_size) -> (batch_size * num_tokens, hidden_size)
            weight (torch.Tensor): (vocab_size, hidden_size)
            labels (torch.Tensor): (batch_size, num_tokens) -> (batch_size * num_tokens, )
            reduction (typing.Optional[str], optional): _description_. Defaults to "mean".
            temperature (typing.Optional[float], optional): _description_. Defaults to 1.0.
            dist_process_group (typing.Optional[dist.ProcessGroup], optional): _description_. Defaults to None.

        Returns:
            typing.List[torch.Tensor]: _description_
        """

        with torch.cuda.nvtx.range("LinearCrossEntropy-forward"):
            REDUCTION = kernels.get_entropy_reduction_enum_number(reduction.lower())

            if len(hidden.shape) != 2:
                hidden = hidden.view(-1, hidden.shape[-1])  # (batch_size * num_tokens, hidden_size)
            if len(labels.shape) != 1:
                labels = labels.view(-1)

            logprobs, entropy, _maximum, _accumulate, _entropy_b = kernels.efficient_entropy_forward(hidden, weight, labels, REDUCTION, temperature, dist_process_group)

            ctx.save_for_backward(hidden, weight, labels, _maximum, _accumulate, _entropy_b)
            ctx.REDUCTION = REDUCTION
            ctx.dist_process_group = dist_process_group
            ctx.should_return_fp32_grad = False
            ctx.temperature = temperature
        return logprobs, entropy

    @staticmethod
    def backward(ctx, dlogprobs: torch.Tensor, dentropy: torch.Tensor) -> typing.List[torch.Tensor]:
        with torch.cuda.nvtx.range("LinearCrossEntropy-backward"):
            (hidden, weight, labels, _maximum, _accumulate, _entropy_b) = ctx.saved_tensors
            REDUCTION = ctx.REDUCTION
            dist_process_group = ctx.dist_process_group
            should_return_fp32_grad = ctx.should_return_fp32_grad
            temperature = ctx.temperature

            d_hidden, d_weight = kernels.efficient_entropy_backward(dlogprobs, dentropy, hidden, weight, labels, _maximum, _accumulate, _entropy_b, REDUCTION, should_return_fp32_grad, temperature, dist_process_group)

        return (d_hidden, d_weight, None, None, None, None)


linear_cross_entropy = LinearCrossEntropy.apply


class TorchEntropyTP(torch.autograd.Function):
    """
    it is used for testing the correctness of the kernel
    it is not efficient and is not recommended to use in practice
    """

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, temperature: float, dist_process_group: torch.distributed.ProcessGroup):
        # weight has shape [vocab_size, hidden_size], hidden has shape [num_tokens, hidden_size]
        logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32).T)  # [num_tokens, vocab_size]
        logits /= temperature
        whole_logits = torch.empty((logits.shape[0], logits.shape[1] * dist.get_world_size(dist_process_group)), dtype=logits.dtype, device=logits.device)
        whole_logits_ref = [whole_logits[:, i * logits.shape[1] : (i + 1) * logits.shape[1]] for i in range(dist.get_world_size(dist_process_group))]
        dist.all_gather(whole_logits_ref, logits, group=dist_process_group)

        pd = torch.nn.functional.softmax(whole_logits, dim=-1)
        entropy_a = torch.logsumexp(whole_logits, dim=-1)  # [num_tokens]
        entropy_b = torch.sum(pd * whole_logits, dim=-1)  # [num_tokens]
        entropy = entropy_a - entropy_b

        logprobs = torch.nn.functional.cross_entropy(whole_logits, labels, reduction="none")
        logprobs = torch.neg(logprobs)

        ctx.save_for_backward(hidden, weight, labels, whole_logits, entropy_b)
        ctx.dist_process_group = dist_process_group
        ctx.temperature = temperature
        return logprobs, entropy

    @staticmethod
    def backward(ctx, g_logprobs: torch.Tensor, g_entropy: torch.Tensor):
        hidden, weight, labels, whole_logits, entropy_b = ctx.saved_tensors
        dist_process_group = ctx.dist_process_group
        temperature = ctx.temperature
        batch_size, hidden_size = hidden.shape
        vocab_size, hidden_size = weight.shape
        rank = dist.get_rank(dist_process_group)

        # Compute softmax probabilities
        maximum, _ = torch.max(whole_logits, dim=-1, keepdim=True)
        exp_logits = torch.exp(whole_logits - maximum)
        accumulate = exp_logits.sum(dim=-1, keepdim=True)
        pd = exp_logits / accumulate

        # Gradient for entropy
        # entropy = entropy_a - entropy_b
        # entropy_a = log(sum(exp(logits)))
        # entropy_b = sum(pd * logits)
        # d_entropy_a/d_logits = pd
        # d_entropy_b/d_logits = pd * (logits - b.unsqueeze(1) + 1)
        # d_entropy/d_logits = d_entropy_a - d_entropy_b
        # d_entropy/d_logits = pd - pd * (logits - b.unsqueeze(1) + 1)
        # d_entropy/d_logits = -pd * (logits - b.unsqueeze(1))
        d_logits_entropy = g_entropy.unsqueeze(1) * (-pd * (whole_logits - entropy_b.unsqueeze(1)))

        # Gradient for logprobs
        # logprobs = -cross_entropy = -log(pd[labels])
        # d_logprobs/d_logits = (pd - one_hot(labels))
        one_hot = torch.zeros_like(whole_logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        g_logprobs = torch.neg(g_logprobs)
        d_logits_logprobs = g_logprobs.unsqueeze(1) * (pd - one_hot)
        # NOTE: This will lead to wrong result
        # d_logits_logprobs = g_logprobs.unsqueeze(1) * (pd - 1) * one_hot

        # Combine gradients
        d_logits = d_logits_entropy + d_logits_logprobs
        d_logits /= temperature

        # Get local slice of gradients
        local_d_logits = d_logits[:, rank * vocab_size : (rank + 1) * vocab_size]

        # Compute gradients for hidden and weight
        d_hidden = torch.matmul(local_d_logits, weight.to(torch.float32))
        d_weight = torch.matmul(local_d_logits.T, hidden.to(torch.float32))

        return d_hidden, d_weight, None, None, None


run_torch_entropy_tp = TorchEntropyTP.apply
