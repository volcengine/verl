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

"""CPU tests for PrimeRewardManager result handling (including dict-returning score functions)."""

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.workers.reward_manager.prime import PrimeRewardManager, run_reward_scoring


class _DummyTokenizer:
    """Minimal tokenizer stub returning deterministic decoded completions."""

    def batch_decode(self, token_ids, skip_special_tokens=True):
        """Return a fixed completion string per sequence in token_ids."""
        return ["the answer is 42" for _ in range(len(token_ids))]


def _dict_compute_score(task, completion, reference, extra_info=None):
    """Return a dict-shaped reward payload matching math_dapo / rlla."""
    return {"score": 1.0, "acc": True, "pred": "42"}


def _tuple_compute_score(task, completion, reference, extra_info=None):
    """Return a tuple-shaped reward payload matching prime_math."""
    return (0.75, {"parsed": "42"})


def _float_compute_score(task, completion, reference, extra_info=None):
    """Return a scalar float reward score."""
    return 0.5


def _make_minimal_data(batch_size=2, prompt_len=2, response_len=3):
    """Construct a minimal DataProto batch for PrimeRewardManager CPU tests."""
    total = prompt_len + response_len
    return DataProto.from_dict(
        tensors={
            "prompts": torch.zeros((batch_size, prompt_len), dtype=torch.long),
            "responses": torch.zeros((batch_size, response_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, total), dtype=torch.long),
        },
        non_tensors={
            "data_source": np.array(["math_dapo"] * batch_size, dtype=object),
            "reward_model": np.array([{"ground_truth": "42"} for _ in range(batch_size)], dtype=object),
        },
    )


def test_run_reward_scoring_supports_dict_tuple_and_float_results():
    """run_reward_scoring must extract result['score'] when the evaluation function returns a dict."""
    dict_scores = run_reward_scoring(
        _dict_compute_score,
        completions=["the answer is 42", "another 42"],
        references=["42", "42"],
        tasks=["math_dapo", "math_dapo"],
        extra_info=[None, None],
        num_processes=1,
    )
    assert dict_scores == [1.0, 1.0]

    tuple_scores = run_reward_scoring(
        _tuple_compute_score,
        completions=["the answer is 42"],
        references=["42"],
        tasks=["prime_math"],
        extra_info=[None],
        num_processes=1,
    )
    assert tuple_scores == [pytest.approx(0.75)]

    float_scores = run_reward_scoring(
        _float_compute_score,
        completions=["the answer is 42"],
        references=["42"],
        tasks=["prime_math"],
        extra_info=[None],
        num_processes=1,
    )
    assert float_scores == [pytest.approx(0.5)]


def test_prime_reward_manager_preserves_dict_score():
    """PrimeRewardManager must not zero out the batch when compute_score returns a dict."""
    manager = PrimeRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=_dict_compute_score,
    )
    data = _make_minimal_data(batch_size=2, prompt_len=2, response_len=3)
    out = manager(data, return_dict=True)

    reward_tensor = out["reward_tensor"]
    assert reward_tensor.shape == (2, 3)
    assert torch.all(reward_tensor[:, -1] == 1.0)
    assert torch.all(reward_tensor[:, :-1] == 0.0)
    assert torch.all(data.batch["acc"] == 1.0)
