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

import torch
import torch.distributed as dist
import typing

try:
    from verl.utils.kernel import linear_cross_entropy
except ImportError:
    # FIXME: remove these manually included paths
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")))
finally:
    from verl.utils.kernel import linear_cross_entropy, set_backward_method, BackwardEnum

import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import logprobs_from_logits

compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)


def run_torch_entropy(hidden: torch.Tensor,
                      weight: torch.Tensor,
                      labels: torch.Tensor,
                      reduction="none") -> typing.List[torch.Tensor]:
    logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32))  # [num_tokens, vocab_size]
    pd = torch.nn.functional.softmax(logits, dim=-1)  # [num_tokens, vocab_size]
    entropy_a = torch.logsumexp(logits, dim=-1)  # [num_tokens]
    entropy_b = torch.sum(pd * logits, dim=-1)  # [num_tokens]
    entropy = entropy_a - entropy_b
    logprobs = torch.nn.functional.cross_entropy(logits, labels, reduction=reduction)  # [num_tokens]
    logprobs = torch.neg(logprobs)
    return logprobs, entropy


class TorchEntropyTP(torch.autograd.Function):
    """
    it is used for testing the correctness of the kernel
    it is not efficient and is not recommended to use in practice
    """

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                dist_process_group: torch.distributed.ProcessGroup):
        logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32))  # [num_tokens, vocab_size]
        whole_logits = torch.empty((logits.shape[0], logits.shape[1] * dist.get_world_size(dist_process_group)),
                                   dtype=logits.dtype,
                                   device=logits.device)
        whole_logits_ref = [
            whole_logits[:, i * logits.shape[1]:(i + 1) * logits.shape[1]]
            for i in range(dist.get_world_size(dist_process_group))
        ]
        dist.all_gather(whole_logits_ref, logits, group=dist_process_group)

        pd = torch.nn.functional.softmax(whole_logits, dim=-1)
        entropy_a = torch.logsumexp(whole_logits, dim=-1)  # [num_tokens]
        entropy_b = torch.sum(pd * whole_logits, dim=-1)  # [num_tokens]
        entropy = entropy_a - entropy_b

        logprobs = torch.nn.functional.cross_entropy(whole_logits, labels, reduction="none")
        logprobs = torch.neg(logprobs)

        ctx.save_for_backward(hidden, weight, labels, whole_logits, entropy_b)
        ctx.dist_process_group = dist_process_group

        return logprobs, entropy

    @staticmethod
    def backward(ctx, g_logprobs: torch.Tensor, g_entropy: torch.Tensor):
        hidden, weight, labels, whole_logits, entropy_b = ctx.saved_tensors
        dist_process_group = ctx.dist_process_group

        batch_size, hidden_size = hidden.shape
        vocab_size = weight.shape[1]
        world_size = dist.get_world_size(dist_process_group)
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

        # Get local slice of gradients
        local_d_logits = d_logits[:, rank * vocab_size:(rank + 1) * vocab_size]

        # Compute gradients for hidden and weight
        d_hidden = torch.matmul(local_d_logits, weight.to(torch.float32).T)
        d_weight = torch.matmul(hidden.to(torch.float32).T, local_d_logits)

        return d_hidden, d_weight, None, None


run_torch_entropy_tp = TorchEntropyTP.apply

def run_verl_actor_entropy(hidden: torch.Tensor,
                           weight: torch.Tensor,
                           labels: torch.Tensor,
                           reduction="none") -> typing.List[torch.Tensor]:
    logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32))  # [num_tokens, vocab_size]
    # compute entropy
    entropy = compute_entropy_from_logits(logits)  # ((total_nnz / sp) + pad)
    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
    logprobs = logprobs_from_logits(logits=logits, labels=labels)
    return logprobs, entropy


class TestLinearCrossEntropy:

    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()
        torch.cuda.synchronize()

    def generate_hyper(self):
        self.num_tokens = 80
        self.hidden_size = 4096
        self.vocab_size = 152064
        self.dtype = torch.bfloat16

    def generate_forward_inputs(self):
        hidden = (torch.empty((self.num_tokens, self.hidden_size), dtype=self.dtype,
                              device="cuda").uniform_(-0.5, 0.5).requires_grad_())
        weight = (torch.empty((self.hidden_size, self.vocab_size), dtype=self.dtype,
                              device="cuda").uniform_(-0.5, 0.5).requires_grad_())
        labels = torch.randint(0, self.vocab_size, (self.num_tokens,), device="cuda")
        return hidden, weight, labels

    def generate_backward_inputs(self):
        g_entropy = (torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-0.5, 0.5))
        g_logprobs = (torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-1, 1))
        return g_entropy, g_logprobs

    def verify_correctness(self, iterations=5):
        self.cleanup()
        self.generate_hyper()

        torch_forward_latency = list()
        torch_backward_latency = list()
        verl_forward_latency = list()
        verl_backward_latency = list()
        kernel_forward_latency = list()
        kernel_backward_latency = list()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i in range(iterations):
            print(f"[INFO]: Iteration {i + 1} / {iterations}...", end='\r')
            hidden, weight, labels = self.generate_forward_inputs()

            start_event.record()
            (torch_logprobs, torch_entropy) = run_torch_entropy(hidden, weight, labels)
            end_event.record()
            torch.cuda.synchronize()
            torch_forward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (verl_logprobs, verl_entropy) = run_verl_actor_entropy(hidden, weight, labels)
            end_event.record()
            torch.cuda.synchronize()
            verl_forward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (kernel_logprobs, kernel_entropy) = linear_cross_entropy(hidden, weight, labels, "none")
            end_event.record()
            torch.cuda.synchronize()
            kernel_forward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(torch_logprobs, verl_logprobs, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(torch_entropy, verl_entropy, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(torch_entropy, kernel_entropy, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(verl_logprobs, kernel_logprobs, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(verl_entropy, kernel_entropy, atol=1e-4, rtol=1e-4)

            # backward
            g_entropy, g_logprobs = self.generate_backward_inputs()

            start_event.record()
            (d_torch_hidden, d_torch_weight) = torch.autograd.grad((torch_entropy, torch_logprobs), (hidden, weight),
                                                                   (g_entropy, g_logprobs),
                                                                   retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            torch_backward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (d_verl_hidden, d_verl_weight) = torch.autograd.grad((verl_entropy, verl_logprobs), (hidden, weight),
                                                                 (g_entropy, g_logprobs),
                                                                 retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            verl_backward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad((kernel_entropy, kernel_logprobs),
                                                                     (hidden, weight), (g_entropy, g_logprobs),
                                                                     retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            kernel_backward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(d_torch_hidden, d_verl_hidden, atol=1e-2, rtol=1e-4)
            torch.testing.assert_close(d_torch_weight, d_verl_weight, atol=1e-2, rtol=1e-4)
            torch.testing.assert_close(d_torch_hidden, d_kernel_hidden, atol=1e-2, rtol=1e-4)
            torch.testing.assert_close(d_torch_weight, d_kernel_weight, atol=1e-2, rtol=1e-4)
            torch.testing.assert_close(d_verl_hidden, d_kernel_hidden, atol=1e-2, rtol=1e-4)
            torch.testing.assert_close(d_verl_weight, d_kernel_weight, atol=1e-2, rtol=1e-4)

        # remove first latency
        torch_forward_latency = torch_forward_latency[1:]
        torch_backward_latency = torch_backward_latency[1:]
        verl_forward_latency = verl_forward_latency[1:]
        verl_backward_latency = verl_backward_latency[1:]
        kernel_forward_latency = kernel_forward_latency[1:]
        kernel_backward_latency = kernel_backward_latency[1:]

        print(f"\n[INFO]: Verified forward & backward correctness.")

        print(f"[INFO]: Forward pass: Torch implementation average time: "
              f"{sum(torch_forward_latency) / len(torch_forward_latency):.2f} ms")
        print(f"[INFO]: Backward pass: torch implementation average time: "
              f"{sum(torch_backward_latency) / len(torch_backward_latency):.2f} ms")
        print(f"[INFO]: Forward pass: VeRL implementation average time: "
              f"{sum(verl_forward_latency) / len(verl_forward_latency):.2f} ms")
        print(f"[INFO]: Backward pass: VeRL implementation average time: "
              f"{sum(verl_backward_latency) / len(verl_backward_latency):.2f} ms")
        print(f"[INFO]: Forward pass: Kernel implementation average time: "
              f"{sum(kernel_forward_latency) / len(kernel_forward_latency):.2f} ms")
        print(f"[INFO]: Backward pass: kernel implementation average time: "
              f"{sum(kernel_backward_latency) / len(kernel_backward_latency):.2f} ms")

    def check_storage(self, method_name, run_forward, reduction="none"):
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()

        torch.cuda.reset_peak_memory_stats()
        (logprobs, entropy) = run_forward(hidden, weight, labels, reduction)
        torch.cuda.synchronize()
        torch_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"[INFO]: {method_name} Forward pass peak memory: {torch_max_memory:.2f} MB")

        g_entropy, g_logprobs = self.generate_backward_inputs()

        torch.cuda.reset_peak_memory_stats()
        (d_torch_hidden, d_torch_weight) = torch.autograd.grad((entropy, logprobs), (hidden, weight),
                                                               (g_entropy, g_logprobs),
                                                               retain_graph=False)
        torch.cuda.synchronize()
        torch_backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"[INFO]: {method_name} Backward pass peak memory: {torch_backward_max_memory:.2f} MB")

    def check_storage_all(self):
        self.check_storage("Torch", run_torch_entropy)
        self.check_storage("VeRL", run_verl_actor_entropy)
        self.check_storage("Kernel", linear_cross_entropy)


class TestLinearCrossEntropy_TensorParallel:

    def __init__(self):
        dist.init_process_group(backend="nccl")
        self.group = dist.group.WORLD

        self.local_rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(device)
        print(f"[INFO]: Local rank: {self.local_rank}, World size: {self.world_size}")

    def shutdown(self):
        dist.destroy_process_group()

    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()
        torch.cuda.synchronize()

    def generate_hyper(self):
        self.num_tokens = 80
        self.hidden_size = 4096
        self.vocab_size = 152064
        self.dtype = torch.bfloat16
        self.iterations = 5

    def generate_forward_inputs(self):
        hidden = (torch.empty((self.num_tokens, self.hidden_size), dtype=self.dtype,
                              device="cuda").uniform_(-0.5, 0.5).requires_grad_())
        weight = (torch.empty((self.hidden_size, self.vocab_size), dtype=self.dtype,
                              device="cuda").uniform_(-0.5, 0.5).requires_grad_())
        labels = torch.randint(0, self.vocab_size * self.world_size, (self.num_tokens,), device="cuda")
        return hidden, weight, labels

    def generate_backward_inputs(self):
        g_entropy = (torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-0.5, 0.5))
        g_logprobs = (torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-1, 1))
        return g_entropy, g_logprobs

    def verify_torch_itself(self):
        self.cleanup()
        self.generate_hyper()

        for i in range(self.iterations):
            hidden, weight, labels = self.generate_forward_inputs()

            # NOTE: we need to manually synchronize hidden and labels among Process Group
            dist.broadcast(hidden, src=0, group=self.group)
            dist.broadcast(labels, src=0, group=self.group)

            # forward pass
            whole_weight = torch.empty((weight.shape[0], weight.shape[1] * self.world_size),
                                       dtype=weight.dtype,
                                       device=weight.device)
            whole_weight_ref = [
                whole_weight[:, i * weight.shape[1]:(i + 1) * weight.shape[1]] for i in range(self.world_size)
            ]
            dist.all_gather(whole_weight_ref, weight, group=self.group)
            whole_weight.requires_grad_()

            (single_logprobs, single_entropy) = run_torch_entropy(hidden, whole_weight, labels)

            (tp_logprobs, tp_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.group)

            torch.testing.assert_close(single_logprobs, tp_logprobs, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(single_entropy, tp_entropy, atol=1e-4, rtol=1e-4)

            # backward pass
            g_entropy, g_logprobs = self.generate_backward_inputs()
            # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
            dist.broadcast(g_entropy, src=0, group=self.group)
            dist.broadcast(g_logprobs, src=0, group=self.group)

            (single_d_hidden, single_d_weight) = torch.autograd.grad((single_entropy, single_logprobs),
                                                                     (hidden, whole_weight), (g_entropy, g_logprobs),
                                                                     retain_graph=False)

            (tp_d_hidden, tp_d_weight) = torch.autograd.grad((tp_entropy, tp_logprobs), (hidden, weight),
                                                             (g_entropy, g_logprobs),
                                                             retain_graph=False)
            # NOTE: all-reduce on hidden is conducted outside the kernel
            dist.all_reduce(tp_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            torch.testing.assert_close(tp_d_hidden, single_d_hidden, atol=1e-2, rtol=1e-4)
            torch.testing.assert_close(tp_d_weight,
                                       single_d_weight[:, self.local_rank * tp_d_weight.shape[1]:(self.local_rank + 1) *
                                                       tp_d_weight.shape[1]])  #,
            # atol=1e-3, rtol=1e-4)
        if self.local_rank == 0:
            print(f"[PASS] torch TP correctness is verified")

    def check_torch_storage(self):
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()

        # NOTE: we need to manually synchronize hidden and labels among Process Group
        dist.broadcast(hidden, src=0, group=self.group)
        dist.broadcast(labels, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (tp_logprobs, tp_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.group)
        torch.cuda.synchronize()
        forward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        g_entropy, g_logprobs = self.generate_backward_inputs()
        # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
        dist.broadcast(g_entropy, src=0, group=self.group)
        dist.broadcast(g_logprobs, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (d_tp_hidden, d_tp_weight) = torch.autograd.grad((tp_entropy, tp_logprobs), (hidden, weight),
                                                         (g_entropy, g_logprobs),
                                                         retain_graph=False)
        torch.cuda.synchronize()
        backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        # NOTE: all-reduce on hidden is conducted outside the kernel
        dist.all_reduce(d_tp_hidden, op=dist.ReduceOp.SUM, group=self.group)

        if self.local_rank == 0:
            print(f"[INFO]: Torch Forward pass peak memory: {forward_max_memory:.2f} MB")
            print(f"[INFO]: Torch Backward pass peak memory: {backward_max_memory:.2f} MB")

    def verify_kernel_correctness(self):
        self.cleanup()
        self.generate_hyper()

        torch_forward_latency = list()
        torch_backward_latency = list()
        kernel_forward_latency = list()
        kernel_backward_latency = list()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i in range(self.iterations):
            hidden, weight, labels = self.generate_forward_inputs()

            # NOTE: we need to manually synchronize hidden and labels among Process Group
            dist.broadcast(hidden, src=0, group=self.group)
            dist.broadcast(labels, src=0, group=self.group)

            start_event.record()
            (torch_logprobs, torch_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.group)
            end_event.record()
            torch.cuda.synchronize()
            torch_forward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (kernel_logprobs, kernel_entropy) = linear_cross_entropy(hidden, weight, labels, "none", self.group)
            end_event.record()
            torch.cuda.synchronize()
            kernel_forward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(torch_entropy, kernel_entropy, atol=1e-4, rtol=1e-4)

            # backward pass
            g_entropy, g_logprobs = self.generate_backward_inputs()
            # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
            dist.broadcast(g_entropy, src=0, group=self.group)
            dist.broadcast(g_logprobs, src=0, group=self.group)

            start_event.record()
            (torch_d_hidden, torch_d_weight) = torch.autograd.grad((torch_entropy, torch_logprobs), (hidden, weight),
                                                                   (g_entropy, g_logprobs),
                                                                   retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            torch_backward_latency.append(start_event.elapsed_time(end_event))
            # NOTE: all-reduce on hidden is conducted outside the kernel
            dist.all_reduce(torch_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            start_event.record()
            (kernel_d_hidden, kernel_d_weight) = torch.autograd.grad((kernel_entropy, kernel_logprobs),
                                                                     (hidden, weight), (g_entropy, g_logprobs),
                                                                     retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            kernel_backward_latency.append(start_event.elapsed_time(end_event))
            # NOTE: all-reduce on hidden is conducted outside the kernel
            dist.all_reduce(kernel_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            torch.testing.assert_close(torch_d_hidden, kernel_d_hidden, atol=1e-2, rtol=1e-4)
            torch.testing.assert_close(torch_d_weight, kernel_d_weight, atol=1e-2, rtol=1e-4)

        # remove first latency
        torch_forward_latency = torch_forward_latency[1:]
        torch_backward_latency = torch_backward_latency[1:]
        kernel_forward_latency = kernel_forward_latency[1:]
        kernel_backward_latency = kernel_backward_latency[1:]

        if self.local_rank == 0:
            print(f"\n[PASS]: Verified kernel forward & backward correctness.")

            print(f"[INFO]: Forward pass: Torch implementation average time: "
                  f"{sum(torch_forward_latency) / len(torch_forward_latency):.2f} ms")
            print(f"[INFO]: Backward pass: torch implementation average time: "
                  f"{sum(torch_backward_latency) / len(torch_backward_latency):.2f} ms")
            print(f"[INFO]: Forward pass: Kernel implementation average time: "
                  f"{sum(kernel_forward_latency) / len(kernel_forward_latency):.2f} ms")
            print(f"[INFO]: Backward pass: kernel implementation average time: "
                  f"{sum(kernel_backward_latency) / len(kernel_backward_latency):.2f} ms")

    def check_kernel_storage(self):
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()

        # NOTE: we need to manually synchronize hidden and labels among Process Group
        dist.broadcast(hidden, src=0, group=self.group)
        dist.broadcast(labels, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (kernel_logprobs, kernel_entropy) = linear_cross_entropy(hidden, weight, labels, "none", self.group)
        torch.cuda.synchronize()
        kernel_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        g_entropy, g_logprobs = self.generate_backward_inputs()
        # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
        dist.broadcast(g_entropy, src=0, group=self.group)
        dist.broadcast(g_logprobs, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad((kernel_entropy, kernel_logprobs), (hidden, weight),
                                                                 (g_entropy, g_logprobs),
                                                                 retain_graph=False)
        torch.cuda.synchronize()
        kernel_backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        # NOTE: all-reduce on hidden is conducted outside the kernel
        dist.all_reduce(d_kernel_hidden, op=dist.ReduceOp.SUM, group=self.group)

        if self.local_rank == 0:
            print(f"[INFO]: Kernel Forward pass peak memory: {kernel_max_memory:.2f} MB")
            print(f"[INFO]: Kernel Backward pass peak memory: {kernel_backward_max_memory:.2f} MB")


if __name__ == "__main__":
    # TP command: torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/kernel/test_linear_cross_entropy.py

    # Check if running with torchrun (distributed mode)
    is_distributed = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        is_distributed = True
    print(f"[INFO]: Running in {'distributed' if is_distributed else 'non-distributed'} mode")
    torch.manual_seed(233376 + int(os.environ.get("RANK", 0)))

    # set_backward_method(BackwardEnum._Total_Fuse_MN)

    if not is_distributed:
        test = TestLinearCrossEntropy()

        test.verify_correctness()
        test.check_storage_all()
    else:
        test = TestLinearCrossEntropy_TensorParallel()

        test.verify_torch_itself()
        test.check_torch_storage()
        test.verify_kernel_correctness()
        test.check_kernel_storage()

        test.shutdown()
