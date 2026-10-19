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
"""CPU coverage for the rollout vocabulary mask on the FSDP engine's eager log-prob path."""

import pytest
import torch
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.vocab_mask import RolloutVocabMask
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

VOCAB_SIZE = 8
SEQ_LENGTHS = (3, 2)
TEMPERATURE = 0.7


class _EngineConfig:
    entropy_checkpointing = False
    entropy_from_logits_with_chunking = False


def _make_engine(vocab_mask):
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    engine.engine_config = _EngineConfig()
    engine.compute_entropy_from_logits = verl_F.entropy_from_logits
    engine.set_rollout_vocab_mask(vocab_mask)
    return engine


def _run_prepare_model_outputs(engine, use_remove_padding, labels):
    """Return (log_probs, entropy, leaf logits of shape (total_nnz, vocab)) from prepare_model_outputs."""
    total_nnz = sum(SEQ_LENGTHS)
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQ_LENGTHS).cumsum(0).tolist()])
    input_ids = torch.nested.nested_tensor_from_jagged(labels.roll(1), offsets=cu_seqlens)
    leaf = torch.randn(total_nnz, VOCAB_SIZE, requires_grad=True)

    output = type("Output", (), {})()
    output_args = {"input_ids_rmpad_rolled": labels, "temperature_is_one": False}
    if use_remove_padding:
        output.logits = (leaf * 1.0).unsqueeze(0)  # non-leaf, like the lm_head output
        output_args.update(temperature_rmpad=torch.full((total_nnz,), TEMPERATURE), pad_size=0)
    else:
        padded = torch.zeros(len(SEQ_LENGTHS), max(SEQ_LENGTHS), VOCAB_SIZE)
        padded = padded.index_put((torch.tensor([0, 0, 0, 1, 1]), torch.tensor([0, 1, 2, 0, 1])), leaf * 1.0)
        output.logits = padded
        output_args["temperature"] = torch.full((len(SEQ_LENGTHS),), TEMPERATURE)

    micro_batch = TensorDict({"input_ids": input_ids}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch, use_remove_padding=use_remove_padding, pad_mode=DatasetPadMode.NO_PADDING, calculate_entropy=True
    )
    model_output = FSDPEngineWithLMHead.prepare_model_outputs(engine, output, output_args, micro_batch, None)
    return model_output["log_probs"].values(), model_output["entropy"].values(), leaf


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_eager_log_probs_and_entropy_use_the_rollout_vocabulary(use_remove_padding):
    torch.manual_seed(0)
    vocab_mask = RolloutVocabMask(vocab_size=VOCAB_SIZE - 2, banned_token_ids=(3,))
    banned = vocab_mask.banned_mask(VOCAB_SIZE)
    labels = torch.tensor([1, 3, 0, 2, 5])  # label 3 is banned
    engine = _make_engine(vocab_mask)

    log_probs, entropy, leaf = _run_prepare_model_outputs(engine, use_remove_padding, labels)

    logits = (leaf.detach() / TEMPERATURE).masked_fill(banned, float("-inf"))
    expected_log_probs = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    allowed_probs = logits[:, ~banned].softmax(-1)
    expected_entropy = -(allowed_probs * allowed_probs.log()).sum(-1)
    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)
    assert torch.isneginf(log_probs[1])

    allowed_labels = ~banned[labels]
    (log_probs[allowed_labels].sum() + entropy.sum()).backward()
    assert torch.isfinite(leaf.grad).all()
    assert torch.equal(leaf.grad[:, banned], torch.zeros_like(leaf.grad[:, banned]))


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_without_mask_the_full_vocabulary_is_used(use_remove_padding):
    torch.manual_seed(0)
    labels = torch.tensor([1, 3, 0, 2, 5])

    log_probs, _, leaf = _run_prepare_model_outputs(_make_engine(None), use_remove_padding, labels)

    expected = (leaf.detach() / TEMPERATURE).log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(log_probs, expected)
