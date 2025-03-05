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

if __name__ == "__main__":
    num_tokens = 80
    hidden_size = 4096
    vocab_size = 152064

    dtype = torch.bfloat16

    enabled_fileds = {
        "forward": {"Torch": True, "Kernel": True},
        "backward": {"Torch": True, "Kernel": True}
    }

    # set_backward_method(BackwardEnum._Total_Separate)

    iterations = 5
    for i in range(iterations):
        print(f"[INFO]: ---------- Iteration {i} starts. ----------")
        with torch.cuda.nvtx.range(f"iteration_{i}"):
            hidden = (torch.empty((num_tokens, hidden_size), dtype=dtype, device="cuda")
                    .uniform_(-0.5, 0.5)
                    .requires_grad_())
            weight = (torch.empty((hidden_size, vocab_size), dtype=dtype, device="cuda")
                    .uniform_(-0.5, 0.5)
                    .requires_grad_())
            labels = torch.randint(0, vocab_size, (num_tokens,), device="cuda")

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            if enabled_fileds["forward"]["Torch"]:
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
                (torch_logprobs, torch_entropy) = run_torch_entropy(hidden, weight, labels)
                end_event.record()
                torch.cuda.synchronize()
                torch_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
                print(f"[INFO]: Forward pass: Torch implementation peak memory: {torch_max_memory:.2f} MB")
                print(f"[INFO]: Forward pass: Torch implementation time: {start_event.elapsed_time(end_event):.2f} ms")

            if enabled_fileds["forward"]["Kernel"]:
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
                (kernel_logprobs, kernel_entropy) = linear_cross_entropy(hidden, weight, labels)
                end_event.record()
                torch.cuda.synchronize()
                kernel_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
                print(f"[INFO]: Forward pass: Kernel implementation peak memory: {kernel_max_memory:.2f} MB")
                print(f"[INFO]: Forward pass: Kernel implementation time: {start_event.elapsed_time(end_event):.2f} ms")

                if enabled_fileds["forward"]["Torch"]:
                    torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-3, rtol=1e-3)
                    torch.testing.assert_close(torch_entropy, kernel_entropy, atol=1e-3, rtol=1e-3)
                    print(f"[INFO]: Forward pass: Kernel implementation passed.")

            if enabled_fileds["backward"]["Torch"] or enabled_fileds["backward"]["Kernel"]:
                g_entropy = (torch.empty((num_tokens,), dtype=dtype, device="cuda")
                            .uniform_(-0.5, 0.5))
                g_logprobs = (torch.empty((), dtype=dtype, device="cuda")
                            .uniform_(-1, 1))

            if enabled_fileds["backward"]["Torch"]:
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
                (d_torch_hidden, d_torch_weight) = torch.autograd.grad((torch_entropy, torch_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
                end_event.record()
                torch.cuda.synchronize()
                torch_backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
                print(f"[INFO]: Backward pass: torch implementation peak memory: {torch_backward_max_memory:.2f} MB")
                print(f"[INFO]: Backward pass: torch gradient time: {start_event.elapsed_time(end_event):.2f} ms")

            if enabled_fileds["backward"]["Kernel"]:
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
                (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad((kernel_entropy, kernel_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
                end_event.record()
                torch.cuda.synchronize()
                kernel_backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
                print(f"[INFO]: Backward pass: kernel implementation peak memory: {kernel_backward_max_memory:.2f} MB")
                print(f"[INFO]: Backward pass: kernel gradient time: {start_event.elapsed_time(end_event):.2f} ms")

                if enabled_fileds["backward"]["Torch"]:
                    torch.testing.assert_close(d_torch_hidden, d_kernel_hidden, atol=1e-2, rtol=1e-3)
                    torch.testing.assert_close(d_torch_weight, d_kernel_weight, atol=1e-2, rtol=1e-3)
                    print(f"[INFO]: Backward pass: kernel implementation passed.")