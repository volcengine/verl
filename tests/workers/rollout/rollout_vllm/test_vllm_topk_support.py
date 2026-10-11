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
"""Under top-p/top-k sampling, vLLM's processed top-k log-probs hold the whole sampling support.

Environment overrides:
  VLLM_TOPK_SUPPORT_MODEL_PATH - model (default Qwen2.5-0.5B-Instruct)
"""

import os

import numpy as np
from vllm import LLM, SamplingParams

from verl.workers.rollout.utils import extract_response_topk_logprobs

MODEL_PATH = os.path.expanduser(os.getenv("VLLM_TOPK_SUPPORT_MODEL_PATH", "~/models/Qwen/Qwen2.5-0.5B-Instruct"))


def test_processed_topk_head_is_the_truncated_support():
    k = 64
    llm = LLM(
        MODEL_PATH,
        logprobs_mode="processed_logprobs",
        max_logprobs=k,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        seed=0,
    )
    params = SamplingParams(temperature=1.0, top_p=0.9, top_k=k, max_tokens=64, logprobs=k, flat_logprobs=True)
    outputs = llm.generate(["Write one short sentence about sampling.", "List three prime numbers."], params)
    support_sizes = []
    for output in outputs:
        completion = output.outputs[0]
        sampled, ids, log_probs = extract_response_topk_logprobs(completion.logprobs, k)
        in_support = np.isfinite(log_probs)
        support_sizes.extend(in_support.sum(-1).tolist())
        # the kept tokens are renormalized over the support, the rest of the head is -inf
        np.testing.assert_allclose(np.where(in_support, np.exp(log_probs), 0.0).sum(-1), 1.0, atol=1e-3)
        for t, token_id in enumerate(completion.token_ids):
            slots = np.nonzero((ids[t] == token_id) & in_support[t])[0]
            assert slots.size == 1, f"sampled token {token_id} is outside the recorded support at position {t}"
            np.testing.assert_allclose(sampled[t], log_probs[t, slots[0]], atol=1e-5)
    assert min(support_sizes) >= 1 and max(support_sizes) <= k
    assert min(support_sizes) < k, "top-p never truncated below the head size, nothing was exercised"
