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

import numpy as np
import pytest

from verl.utils.reward_score.search_r1_like_qa_em import (
    compute_score,
    compute_score_subem,
    em_check,
    subem_check,
)


def _answer(text):
    return f"<think>...</think><answer>{text}</answer>"


@pytest.mark.parametrize(
    "prediction",
    ["Paris", "the city of Paris", "It is Paris, France.", "PARIS!"],
)
def test_subem_matches_whole_words(prediction):
    assert subem_check(prediction, ["Paris"]) == 1


@pytest.mark.parametrize(
    "prediction, golden",
    [
        ("comparison", "Paris"),  # "paris" is only a substring of a word
        ("Parisa Khosravi", "Paris"),
        ("1999", "1"),
        ("Newcastle", "New"),
    ],
)
def test_subem_rejects_partial_words(prediction, golden):
    assert subem_check(prediction, [golden]) == 0


def test_multi_word_golden_answer_must_be_contiguous():
    assert subem_check("born in New York City", ["New York"]) == 1
    assert subem_check("New ideas from York", ["New York"]) == 0


def test_golden_answer_that_normalizes_to_empty_matches_nothing():
    # "The" and "." normalize to "", which used to be a substring of (or equal to) everything
    assert subem_check("anything at all", ["The"]) == 0
    assert em_check(".", ["The"]) == 0
    assert subem_check("Paris", [".", "Paris"]) == 1


def test_golden_answers_as_numpy_array():
    assert subem_check("Paris", np.array(["Lyon", "Paris"])) == 1
    assert em_check("paris", np.array(["Paris"])) == 1


def test_compute_score_subem():
    ground_truth = {"target": ["Paris"]}
    assert compute_score_subem(_answer("Paris"), ground_truth) == 1.0
    assert compute_score_subem(_answer("comparison"), ground_truth) == 0.0
    assert compute_score_subem("no answer tag", ground_truth) == 0


def test_max_answer_tokens_blocks_listing_candidates():
    ground_truth = {"target": ["Paris"]}
    dump = _answer("London Berlin Rome Madrid Paris Tokyo")
    assert compute_score_subem(dump, ground_truth) == 1.0  # default: no limit, unchanged behaviour
    assert compute_score_subem(dump, ground_truth, max_answer_tokens=3) == 0.0
    assert compute_score_subem(_answer("Paris, France"), ground_truth, max_answer_tokens=3) == 1.0


def test_compute_score_em_unchanged():
    ground_truth = {"target": ["Paris"]}
    assert compute_score(_answer("The Paris"), ground_truth) == 1.0
    assert compute_score(_answer("Paris France"), ground_truth) == 0.0
