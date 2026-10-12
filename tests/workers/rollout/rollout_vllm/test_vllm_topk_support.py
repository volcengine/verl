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
"""Check provider evidence for complete and incomplete vLLM sampler heads.

Environment overrides:
  VLLM_TOPK_SUPPORT_MODEL_PATH - model (default Qwen2.5-0.5B-Instruct)
"""

import os

import numpy as np
import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.logprobs import FlatLogprobs, append_logprobs_for_next_position
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

from verl.workers.rollout.utils import extract_response_topk_logprobs_with_coverage

MODEL_PATH = os.path.expanduser(os.getenv("VLLM_TOPK_SUPPORT_MODEL_PATH", "~/models/Qwen/Qwen2.5-0.5B-Instruct"))


@pytest.fixture(scope="module")
def llm():
    k = 64
    return LLM(
        MODEL_PATH,
        logprobs_mode="processed_logprobs",
        max_logprobs=k,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        seed=0,
    )


@pytest.mark.parametrize("top_k,top_p", [(16, 1.0), (64, 0.9), (-1, 0.9), (-1, 1.0)])
def test_processed_topk_support_evidence(llm, top_k, top_p):
    k = 64
    params = SamplingParams(temperature=1.0, top_p=top_p, top_k=top_k, max_tokens=32, logprobs=k, flat_logprobs=True)
    outputs = llm.generate(["Write one short sentence about sampling.", "List three prime numbers."], params)
    complete_rows = 0
    for output in outputs:
        completion = output.outputs[0]
        sampled, ids, log_probs, complete = extract_response_topk_logprobs_with_coverage(
            completion.logprobs, k, vocab_size=llm.llm_engine.model_config.get_vocab_size()
        )
        in_support = np.isfinite(log_probs)
        # A partial head with all finite entries is unknown, even if its mass rounds to one.
        np.testing.assert_array_equal(complete, in_support.sum(-1) < k)
        complete_rows += int(complete.sum())
        np.testing.assert_allclose(np.exp(log_probs[complete]).sum(-1), 1.0, atol=1e-3)
        for t, token_id in enumerate(completion.token_ids):
            slots = np.nonzero((ids[t] == token_id) & in_support[t])[0]
            if complete[t]:
                assert slots.size == 1, f"sampled token {token_id} is outside a proven complete head at position {t}"
            if slots.size:
                np.testing.assert_allclose(sampled[t], log_probs[t, slots[0]], atol=1e-5)
    if top_k == 16:
        assert complete_rows > 0
    if top_k == -1 and top_p == 1.0:
        assert complete_rows == 0


@pytest.mark.parametrize("k,expected_complete", [(2, False), (4, True)])
def test_small_decode_batch_ties_do_not_prove_support_fits_configured_top_k(k, expected_complete):
    # The small-batch PyTorch provider retains both cutoff ties despite top_k=2. Record the
    # omitted tied token as the sampled token; it has positive probability in the real sampler.
    logits = torch.tensor([[2.0, 1.0, 1.0, 0.0]], device="cuda")
    logits = apply_top_k_top_p(logits, torch.tensor([2], device="cuda"), None)
    log_probs = logits.log_softmax(-1)
    assert int(log_probs.isfinite().sum()) == 3
    head, ids = log_probs.topk(k, -1)
    sampled_id = 2
    flat = FlatLogprobs()
    append_logprobs_for_next_position(
        flat,
        [sampled_id, *ids[0].tolist()],
        [log_probs[0, sampled_id].item(), *head[0].tolist()],
        [None] * (k + 1),
        2,
        k,
    )
    _, head_ids, _, complete = extract_response_topk_logprobs_with_coverage(flat, k, vocab_size=4)
    assert complete.tolist() == [expected_complete]
    if k == 2:
        assert not (head_ids == sampled_id).any()
