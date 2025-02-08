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

from verl.utils.model import create_random_mask
from flash_attn.bert_padding import unpad_input
import torch
import pytest


def test_log_probs_from_logits_response_rmpad():
    from verl.utils.torch_functional import log_probs_from_logits_response, log_probs_from_logits_response_rmpad
    vocab_size = 32000
    batch_size = 2
    prompt_length = 256
    response_length = 256

    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, prompt_length + response_length), device='cuda')
    attention_mask = create_random_mask(input_ids=input_ids,
                                        max_ratio_of_left_padding=0.2,
                                        max_ratio_of_valid_token=0.8,
                                        min_ratio_of_valid_token=0.6)

    response_mask = attention_mask[:, -response_length:]

    assert torch.all(response_mask[:, 0] == 1)

    logits = torch.randn(batch_size, prompt_length + response_length, vocab_size, device='cuda')
    logits_rmpad = unpad_input(logits, attention_mask)[0]

    expected_output = log_probs_from_logits_response(input_ids=input_ids,
                                                     logits=logits,
                                                     response_length=response_length)
    actual_output = log_probs_from_logits_response_rmpad(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         logits_rmpad=logits_rmpad,
                                                         response_length=response_length)

    # This should bitwise align as only this operation only contains gather operators
    assert torch.all(torch.eq(actual_output * response_mask, expected_output * response_mask))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_logprobs_from_logits_v2(dtype):
    from verl.utils.torch_functional import logprobs_from_logits_v2, logprobs_from_logits_naive
    vocab_size = 32000
    batch_size = 2
    seq_len = 512

    labels = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device='cuda')
    logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda', dtype=dtype)

    expected_output = logprobs_from_logits_naive(labels=labels, logits=logits)
    actual_output = logprobs_from_logits_v2(labels=labels, logits=logits)

    if dtype in [torch.float16, torch.bfloat16]:  # float16 falls back to an exactly equivalent method
        assert torch.equal(actual_output, expected_output)
    else:  # small numerical difference when using gather / logsumexp approach
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-5)


def test_lr_scheduler():
    from torch import nn
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    from verl.utils.torch_functional import get_constant_schedule_with_warmup
    constant_lr = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=2)

    lr_lst = []

    for _ in range(5):
        lr_lst.append(constant_lr.get_last_lr()[0])
        constant_lr.step()

    torch.testing.assert_close(lr_lst, [0.0, 0.0005, 0.001, 0.001, 0.001])

    from verl.utils.torch_functional import get_cosine_schedule_with_warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    cosine_lr = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=2,
                                                num_training_steps=5,
                                                min_lr_ratio=0.1)

    lr_lst = []

    for _ in range(5):
        lr_lst.append(cosine_lr.get_last_lr()[0])
        cosine_lr.step()

    torch.testing.assert_close(lr_lst, [0.0, 0.0005, 0.001, 0.0007750000000000002, 0.0003250000000000002])
