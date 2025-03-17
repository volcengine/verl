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

def run_torch_entropy(hidden: torch.Tensor,
                    weight: torch.Tensor,
                    labels: torch.Tensor) -> typing.List[torch.Tensor]:
    logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32)) # [num_tokens, vocab_size]
    pd = torch.nn.functional.softmax(logits, dim=-1) # [num_tokens, vocab_size]
    entropy_a = torch.logsumexp(logits, dim=-1) # [num_tokens]
    entropy_b = torch.sum(pd * logits, dim=-1) # [num_tokens]
    entropy = entropy_a - entropy_b
    logprobs = torch.nn.functional.cross_entropy(logits, labels) # [1]
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
        hidden = (torch.empty((self.num_tokens, self.hidden_size), dtype=self.dtype, device="cuda")
                .uniform_(-0.5, 0.5)
                .requires_grad_())
        weight = (torch.empty((self.hidden_size, self.vocab_size), dtype=self.dtype, device="cuda")
                .uniform_(-0.5, 0.5)
                .requires_grad_())
        labels = torch.randint(0, self.vocab_size, (self.num_tokens,), device="cuda")
        return hidden, weight, labels

    def generate_backward_inputs(self):
        g_entropy = (torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda")
                            .uniform_(-0.5, 0.5))
        g_logprobs = (torch.empty((), dtype=self.dtype, device="cuda")
                            .uniform_(-1, 1))
        return g_entropy, g_logprobs

    def verify_correctness(self):
        self.cleanup()
        self.generate_hyper()

        iterations = 5

        torch_forward_latency = list()
        torch_backward_latency = list()
        kernel_forward_latency = list()
        kernel_backward_latency = list()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i in range(iterations):
            hidden, weight, labels = self.generate_forward_inputs()

            start_event.record()
            (torch_logprobs, torch_entropy) = run_torch_entropy(hidden, weight, labels)
            end_event.record()
            torch.cuda.synchronize()
            torch_forward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (kernel_logprobs, kernel_entropy) = linear_cross_entropy(hidden, weight, labels)
            end_event.record()
            torch.cuda.synchronize()
            kernel_forward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(torch_entropy, kernel_entropy, atol=1e-4, rtol=1e-4)

            # backward
            g_entropy, g_logprobs = self.generate_backward_inputs()

            start_event.record()
            (d_torch_hidden, d_torch_weight) = torch.autograd.grad((torch_entropy, torch_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            torch_backward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad((kernel_entropy, kernel_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            kernel_backward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(d_torch_hidden, d_kernel_hidden, atol=1e-2, rtol=1e-4)
            torch.testing.assert_close(d_torch_weight, d_kernel_weight, atol=1e-2, rtol=1e-4)

        # remove first latency
        torch_forward_latency = torch_forward_latency[1:]
        torch_backward_latency = torch_backward_latency[1:]
        kernel_forward_latency = kernel_forward_latency[1:]
        kernel_backward_latency = kernel_backward_latency[1:]

        print(f"[INFO]: Verified forward & backward correctness.")

        print(f"[INFO]: Forward pass: Torch implementation average time: "
              f"{sum(torch_forward_latency) / len(torch_forward_latency):.2f} ms")
        print(f"[INFO]: Backward pass: torch implementation average time: "
              f"{sum(torch_backward_latency) / len(torch_backward_latency):.2f} ms")
        print(f"[INFO]: Forward pass: Kernel implementation average time: "
              f"{sum(kernel_forward_latency) / len(kernel_forward_latency):.2f} ms")
        print(f"[INFO]: Backward pass: kernel implementation average time: "
              f"{sum(kernel_backward_latency) / len(kernel_backward_latency):.2f} ms")


    def check_torch_storage(self):
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()

        torch.cuda.reset_peak_memory_stats()
        (torch_logprobs, torch_entropy) = run_torch_entropy(hidden, weight, labels)
        torch.cuda.synchronize()
        torch_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Torch Forward pass peak memory: {torch_max_memory:.2f} MB")

        g_entropy, g_logprobs = self.generate_backward_inputs()

        torch.cuda.reset_peak_memory_stats()
        (d_torch_hidden, d_torch_weight) = torch.autograd.grad((torch_entropy, torch_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
        torch.cuda.synchronize()
        torch_backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Torch Backward pass peak memory: {torch_backward_max_memory:.2f} MB")

    def check_kernel_storage(self):
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()

        torch.cuda.reset_peak_memory_stats()
        (kernel_logprobs, kernel_entropy) = linear_cross_entropy(hidden, weight, labels)
        torch.cuda.synchronize()
        kernel_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Kernel Forward pass peak memory: {kernel_max_memory:.2f} MB")   

        g_entropy, g_logprobs = self.generate_backward_inputs()

        torch.cuda.reset_peak_memory_stats()
        (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad((kernel_entropy, kernel_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
        torch.cuda.synchronize()
        kernel_backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Kernel Backward pass peak memory: {kernel_backward_max_memory:.2f} MB")


if __name__ == "__main__":
    test = TestLinearCrossEntropy()

    test.verify_correctness()
    test.check_torch_storage()
    test.check_kernel_storage()