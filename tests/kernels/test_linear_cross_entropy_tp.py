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

import os
import typing

import torch
import torch.distributed as dist

try:
    from verl.utils.kernel import linear_cross_entropy, run_torch_entropy_tp
except ImportError:
    # FIXME: remove these manually included paths
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")))
finally:
    from verl.utils.kernel import linear_cross_entropy

import verl.utils.torch_functional as verl_F
from verl.utils.torch_functional import logprobs_from_logits

compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)


def run_torch_entropy(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, temperature: float, reduction="none") -> typing.List[torch.Tensor]:
    # [num_tokens, vocab_size]
    logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32) if weight.size(0) == hidden.size(1) else weight.T.to(torch.float32))
    logits /= temperature
    pd = torch.nn.functional.softmax(logits, dim=-1)  # [num_tokens, vocab_size]
    entropy_a = torch.logsumexp(logits, dim=-1)  # [num_tokens]
    entropy_b = torch.sum(pd * logits, dim=-1)  # [num_tokens]
    entropy = entropy_a - entropy_b
    logprobs = torch.nn.functional.cross_entropy(logits, labels, reduction=reduction)  # [num_tokens]
    logprobs = torch.neg(logprobs)
    return logprobs, entropy


def run_verl_actor_entropy(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, temperature: float, reduction="none") -> typing.List[torch.Tensor]:
    # [num_tokens, vocab_size]
    logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32) if weight.size(0) == hidden.size(1) else weight.T.to(torch.float32))
    logits /= temperature
    # compute entropy
    entropy = compute_entropy_from_logits(logits)  # ((total_nnz / sp) + pad)
    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
    logprobs = logprobs_from_logits(logits=logits, labels=labels)
    return logprobs, entropy


MAX_TEST_CASES = os.environ.get("MAX_TEST_CASES", 5)


class TestLinearCrossEntropy_TensorParallel:
    def __init__(self, test_case_idx: int):
        dist.init_process_group(backend="nccl")
        self.group = dist.group.WORLD

        self.local_rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(device)
        print(f"[INFO]: Local rank: {self.local_rank}, World size: {self.world_size}")
        self.test_case_idx = test_case_idx

    def shutdown(self):
        dist.destroy_process_group()

    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.synchronize()

    def generate_hyper(self):
        self.dtype = torch.bfloat16
        if self.test_case_idx == 0:
            self.batch_size = 1
            self.num_tokens = 1937
            self.hidden_size = 3584
            self.vocab_size = 152064
        elif self.test_case_idx == 1:
            self.batch_size = 1
            self.num_tokens = 2169
            self.hidden_size = 896
            self.vocab_size = 151936
        elif self.test_case_idx == 2:
            self.batch_size = 1
            self.num_tokens = 1530
            self.hidden_size = 2048
            self.vocab_size = 32256
        elif self.test_case_idx == 3:
            self.batch_size = 1
            self.num_tokens = 1388
            self.hidden_size = 4096
            self.vocab_size = 102400
        elif self.test_case_idx == 4:
            self.batch_size = 1
            self.num_tokens = 8192
            self.hidden_size = 4096
            self.vocab_size = 102400
        else:
            raise ValueError(f"Invalid test case index: {self.test_case_idx}")

    def generate_forward_inputs(self):
        hidden = torch.empty((self.batch_size, self.num_tokens, self.hidden_size), dtype=self.dtype, device="cuda").uniform_(-0.5, 0.5).requires_grad_()
        weight = torch.empty((self.vocab_size, self.hidden_size), dtype=self.dtype, device="cuda").uniform_(-0.5, 0.5).requires_grad_()
        labels = torch.randint(0, self.vocab_size, (self.batch_size, self.num_tokens), device="cuda")
        return hidden, weight, labels

    def generate_backward_inputs(self):
        g_entropy = torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-0.5, 0.5)
        g_logprobs = torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-1, 1)
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
            # Create a tensor to hold the gathered weights from all ranks
            # weight has shape [vocab_size, hidden_size]
            # We want to gather along the first dimension to get [vocab_size * world_size, hidden_size]

            # Create a single contiguous tensor to hold all gathered weights
            whole_weight = torch.empty((self.vocab_size * self.world_size, self.hidden_size), dtype=weight.dtype, device=weight.device)

            # Create views into the tensor for each rank's portion
            whole_weight_views = [whole_weight[i * self.vocab_size : (i + 1) * self.vocab_size] for i in range(self.world_size)]

            # Perform all_gather operation using the views
            dist.all_gather(whole_weight_views, weight, group=self.group)

            # Set requires_grad for autograd
            whole_weight.requires_grad_()

            (single_logprobs, single_entropy) = run_torch_entropy(hidden, whole_weight, labels, self.temperature)

            (tp_logprobs, tp_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.temperature, self.group)

            torch.testing.assert_close(single_logprobs, tp_logprobs, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(single_entropy, tp_entropy, atol=1e-4, rtol=1e-4)

            # backward pass
            g_entropy, g_logprobs = self.generate_backward_inputs()
            # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
            dist.broadcast(g_entropy, src=0, group=self.group)
            dist.broadcast(g_logprobs, src=0, group=self.group)

            (single_d_hidden, single_d_weight) = torch.autograd.grad((single_entropy, single_logprobs), (hidden, whole_weight), (g_entropy, g_logprobs), retain_graph=False)

            (tp_d_hidden, tp_d_weight) = torch.autograd.grad((tp_entropy, tp_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False)
            # NOTE: all-reduce on hidden is conducted outside the kernel
            dist.all_reduce(tp_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            torch.testing.assert_close(tp_d_hidden, single_d_hidden, atol=1e-2, rtol=1e-4)
            # Extract the corresponding slice from single_d_weight for comparison
            # tp_d_weight has shape [vocab_size, hidden_size]
            # single_d_weight has shape [vocab_size * world_size, hidden_size]
            torch.testing.assert_close(tp_d_weight, single_d_weight[self.local_rank * self.vocab_size : (self.local_rank + 1) * self.vocab_size], atol=1e-2, rtol=1e-4)

            # atol=1e-3, rtol=1e-4)
        if self.local_rank == 0:
            print("[PASS] torch TP correctness is verified")

    def check_torch_storage(self):
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()

        # NOTE: we need to manually synchronize hidden and labels among Process Group
        dist.broadcast(hidden, src=0, group=self.group)
        dist.broadcast(labels, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (tp_logprobs, tp_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.temperature, self.group)
        torch.cuda.synchronize()
        forward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        g_entropy, g_logprobs = self.generate_backward_inputs()
        # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
        dist.broadcast(g_entropy, src=0, group=self.group)
        dist.broadcast(g_logprobs, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (d_tp_hidden, d_tp_weight) = torch.autograd.grad((tp_entropy, tp_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False)
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
            (torch_logprobs, torch_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.temperature, self.group)
            end_event.record()
            torch.cuda.synchronize()
            torch_forward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (kernel_logprobs, kernel_entropy) = linear_cross_entropy(hidden, weight, labels, "none", self.temperature, self.group)
            end_event.record()
            torch.cuda.synchronize()
            kernel_forward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-1, rtol=1e-2)
            torch.testing.assert_close(torch_entropy, kernel_entropy, atol=1e-1, rtol=1e-2)

            # backward pass
            g_entropy, g_logprobs = self.generate_backward_inputs()
            # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
            dist.broadcast(g_entropy, src=0, group=self.group)
            dist.broadcast(g_logprobs, src=0, group=self.group)

            start_event.record()
            (torch_d_hidden, torch_d_weight) = torch.autograd.grad((torch_entropy, torch_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            torch_backward_latency.append(start_event.elapsed_time(end_event))
            # NOTE: all-reduce on hidden is conducted outside the kernel
            dist.all_reduce(torch_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            start_event.record()
            (kernel_d_hidden, kernel_d_weight) = torch.autograd.grad((kernel_entropy, kernel_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False)
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
            print("\n[PASS]: Verified kernel forward & backward correctness.")

            print(f"[INFO]: Forward pass: Torch implementation average time: {sum(torch_forward_latency) / len(torch_forward_latency):.2f} ms")
            print(f"[INFO]: Backward pass: torch implementation average time: {sum(torch_backward_latency) / len(torch_backward_latency):.2f} ms")
            print(f"[INFO]: Forward pass: Kernel implementation average time: {sum(kernel_forward_latency) / len(kernel_forward_latency):.2f} ms")
            print(f"[INFO]: Backward pass: kernel implementation average time: {sum(kernel_backward_latency) / len(kernel_backward_latency):.2f} ms")

    def check_kernel_storage(self):
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()

        # NOTE: we need to manually synchronize hidden and labels among Process Group
        dist.broadcast(hidden, src=0, group=self.group)
        dist.broadcast(labels, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (kernel_logprobs, kernel_entropy) = linear_cross_entropy(hidden, weight, labels, "none", self.temperature, self.group)
        torch.cuda.synchronize()
        kernel_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        g_entropy, g_logprobs = self.generate_backward_inputs()
        # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
        dist.broadcast(g_entropy, src=0, group=self.group)
        dist.broadcast(g_logprobs, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad((kernel_entropy, kernel_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False)
        torch.cuda.synchronize()
        kernel_backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        # NOTE: all-reduce on hidden is conducted outside the kernel
        dist.all_reduce(d_kernel_hidden, op=dist.ReduceOp.SUM, group=self.group)

        if self.local_rank == 0:
            print(f"[INFO]: Kernel Forward pass peak memory: {kernel_max_memory:.2f} MB")
            print(f"[INFO]: Kernel Backward pass peak memory: {kernel_backward_max_memory:.2f} MB")


if __name__ == "__main__":
    # TP command: torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/kernel/test_linear_cross_entropy_tp.py

    # Check if running with torchrun (distributed mode)
    assert int(os.environ["WORLD_SIZE"]) > 1, "[ERROR]: This test is designed to run in distributed mode with torchrun. Please use torchrun to execute this script."
    torch.manual_seed(233376 + int(os.environ.get("RANK", 0)))

    # set_backward_method(BackwardEnum._Total_Fuse_MN)
    # set_backward_method(BackwardEnum._Split_Dlogits_N)

    for test_case_idx in range(MAX_TEST_CASES):
        test = TestLinearCrossEntropy_TensorParallel(test_case_idx)

        test.verify_torch_itself()
        test.check_torch_storage()
        test.verify_kernel_correctness()
        test.check_kernel_storage()

        test.shutdown()
