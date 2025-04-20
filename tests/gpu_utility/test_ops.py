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


def test_flash_attn_cross_entropy():
    import torch
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    from torch import nn

    from verl.utils.debug import log_gpu_memory_usage
    from verl.utils.torch_functional import logprobs_from_logits_naive

    log_gpu_memory_usage("At start")

    hidden_states = torch.randn(size=(2048, 5120), device="cuda", requires_grad=True, dtype=torch.bfloat16)

    linear = nn.Linear(in_features=5120, out_features=155136, bias=False, device="cuda", dtype=torch.bfloat16)

    logits = linear(hidden_states)

    # logits = logits.float()
    labels = torch.randint(low=0, high=155136, size=(2048,), device="cuda")

    log_gpu_memory_usage("before computation")
    # output = checkpoint.checkpoint(logprobs_from_logits, logits, labels, use_reentrant=True)
    output = -cross_entropy_loss(logits, labels)[0]
    # output = logprobs_from_logits(logits, labels)
    log_gpu_memory_usage("After forward")
    output.sum().backward()
    log_gpu_memory_usage("After backward")

    groundtruth = logprobs_from_logits_naive(logits.float(), labels)

    torch.testing.assert_close(output, groundtruth)
