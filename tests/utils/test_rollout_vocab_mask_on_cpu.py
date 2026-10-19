# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
import torch

import verl.utils.torch_functional as verl_F
from verl.utils.vocab_mask import RolloutVocabMask
from verl.workers.rollout.utils import get_rollout_vocab_mask


class _Tokenizer:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class _Processor:
    image_token_id = 7
    video_token_id = 8


def test_banned_mask_covers_tail_and_banned_ids():
    mask = RolloutVocabMask(vocab_size=10, banned_token_ids=(2, 7)).banned_mask(12)

    expected = torch.zeros(12, dtype=torch.bool)
    expected[[2, 7, 10, 11]] = True
    assert torch.equal(mask, expected)


def test_banned_mask_is_none_without_excluded_ids():
    assert RolloutVocabMask(vocab_size=12).banned_mask(12) is None


def test_banned_mask_rejects_ids_outside_the_logits():
    with pytest.raises(ValueError, match="lie outside"):
        RolloutVocabMask(vocab_size=10, banned_token_ids=(12,)).banned_mask(12)


def test_vllm_rollout_vocab_mask_matches_the_server_arguments():
    vocab_mask = get_rollout_vocab_mask("vllm", _Tokenizer(10), _Processor())

    assert vocab_mask == RolloutVocabMask(vocab_size=10, banned_token_ids=(7, 8))


def test_rollout_without_vocab_mask():
    assert get_rollout_vocab_mask("sglang", _Tokenizer(10), _Processor()) is None


@pytest.mark.parametrize(
    "entropy_fn", [verl_F.entropy_from_logits, lambda logits: verl_F.entropy_from_logits_with_chunking(logits, 2)]
)
def test_entropy_of_masked_logits_is_the_entropy_of_the_renormalized_distribution(entropy_fn):
    torch.manual_seed(0)
    banned = torch.zeros(9, dtype=torch.bool)
    banned[[1, 8]] = True
    logits = torch.randn(5, 9).masked_fill(banned, float("-inf")).requires_grad_()

    entropy = entropy_fn(logits)
    entropy.sum().backward()

    allowed_probs = logits.detach()[:, ~banned].softmax(dim=-1)
    expected = -(allowed_probs * allowed_probs.log()).sum(dim=-1)
    torch.testing.assert_close(entropy, expected)
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[:, banned], torch.zeros(5, 2))
