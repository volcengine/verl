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

# Unless marked as synthetic, ground truths are taken from rows of the PRIME-RL/Eurus-2-RL-Data
# validation split (row index in the comment).

import pytest

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.prime_math.grader import math_equal

_ROW_302 = "\\left[\\frac{3}{2}, +\\infty\\right)"
_ROW_550 = "\\left(0, \\frac{3}{5}\\right) \\cup \\left(1, +\\infty\\right)"
_ROW_762 = "\\left[\\frac{25}{2},\\infty\\right)"


def _score(data_source, answer, ground_truth):
    return default_compute_score(data_source, f"Step 1: solve.\n\nThe answer is $\\boxed{{{answer}}}$.", ground_truth)


# cases that were graded wrongly before this fix
@pytest.mark.parametrize(
    ("data_source", "ground_truth", "answer", "expected"),
    [
        # interval with a wrong endpoint bracket
        ("numina_olympiads", "(-\\infty, \\frac{1}{2}]", "(-\\infty, \\frac{1}{2})", 0.0),  # row 85
        ("numina_cn_k12", "[1,2)", "[1,2]", 0.0),  # row 176
        ("numina_cn_k12", "(0,4]", "(0,4)", 0.0),  # row 214
        ("numina_cn_k12", "(-\\infty,8]", "(-\\infty,8)", 0.0),  # row 990
        # the bracket at an infinite endpoint does not matter
        ("numina_cn_k12", "(-\\infty,8]", "[-\\infty,8]", 1.0),  # row 990
        ("numina_cn_k12", _ROW_302, _ROW_302.replace("\\right)", "\\right]"), 1.0),  # row 302
        ("numina_cn_k12", _ROW_550, _ROW_550.replace("\\infty\\right)", "\\infty\\right]"), 1.0),  # row 550
        ("numina_synthetic_math", _ROW_762, _ROW_762.replace("\\right)", "\\right]"), 1.0),  # row 762
        ("numina_cn_k12", _ROW_302, "[\\frac{3}{2}, +\\infty]", 1.0),  # row 302, without \left / \right
        # unicode infinity
        ("numina_cn_k12", "(-∞, 0)", "(-\\infty, 0)", 1.0),  # row 676
        # 100x, 1/100x and off-by-one numbers
        ("numina_synthetic_math", "22500", "2250000", 0.0),  # row 0
        ("numina_synthetic_math", "22500", "225", 0.0),  # row 0
        ("numina_synthetic_math", "42", "4200", 0.0),  # row 4
        ("numina_synthetic_math", "42", "0.42", 0.0),  # row 4
        ("numina_synthetic_math", "22500", "22501", 0.0),  # row 0
        ("numina_olympiads", "6990500", "6990501", 0.0),  # row 819
        # integers above 2**53, written as integers or with a zero fraction
        ("numina_synthetic_math", "9007199254740992", "9007199254740993", 0.0),  # synthetic
        ("numina_synthetic_math", "9007199254740992", "9007199254740993.0", 0.0),  # synthetic
        ("numina_synthetic_math", "9007199254740992", "+9007199254740993", 0.0),  # synthetic
        ("numina_synthetic_math", "9007199254740992", "+9007199254740993.0", 0.0),  # synthetic
        ("numina_synthetic_math", "-9007199254740992", "-9007199254740993", 0.0),  # synthetic
        ("numina_synthetic_math", "-9007199254740992", "-9007199254740993.0", 0.0),  # synthetic
        # the percent conversion follows the side that carries the percent sign
        ("numina_olympiads", "83\\%", "8300", 0.0),  # row 3
        ("numina_olympiads", "83\\%", "0.83\\%", 0.0),  # row 3
    ],
)
def test_fixed_grading(data_source, ground_truth, answer, expected):
    assert _score(data_source, answer, ground_truth) == expected


# grading that this fix keeps as it was
@pytest.mark.parametrize(
    ("data_source", "ground_truth", "answer", "expected"),
    [
        # tuples
        ("numina_cn_k12", "(1,2)", "(1, 2)", 1.0),  # row 28
        ("numina_cn_k12", "(1,2)", "(2,1)", 0.0),  # row 28
        ("numina_cn_k12", "(1,2)", "[1,2]", 0.0),  # row 28
        ("numina_synthetic_math", "(-8, 7)", "(-8, 8)", 0.0),  # row 297
        # sets
        ("numina_cn_k12", "\\{0, 1, 2\\}", "\\{0,1,2\\}", 1.0),  # row 473
        ("numina_cn_k12", "\\{0, 1, 2\\}", "\\{0, 1, 3\\}", 0.0),  # row 473
        # intervals
        ("numina_cn_k12", "[1,2)", "[1,2)", 1.0),  # row 176
        ("numina_cn_k12", "[1,2)", "(1,2)", 0.0),  # row 176
        ("numina_olympiads", "(-\\infty, \\frac{1}{2}]", "(-\\infty, \\frac{1}{2}]", 1.0),  # row 85
        ("numina_synthetic_math", "(0, \\infty]", "(0, \\infty)", 1.0),  # row 77, bracket at infinity
        ("numina_synthetic_math", "(0, \\infty]", "(0, \\infty]", 1.0),  # row 77
        ("numina_synthetic_math", "(0, \\infty]", "[0, \\infty)", 0.0),  # row 77
        ("numina_cn_k12", _ROW_302, _ROW_302, 1.0),  # row 302
        ("numina_cn_k12", _ROW_302, _ROW_302.replace("\\left[", "\\left("), 0.0),  # row 302, finite endpoint
        ("numina_cn_k12", _ROW_302, "(\\frac{3}{2}, +\\infty)", 0.0),  # row 302, finite endpoint
        ("numina_cn_k12", _ROW_550, _ROW_550.replace("\\left(1", "\\left[1"), 0.0),  # row 550, finite endpoint
        ("numina_cn_k12", "(-∞, 0)", "(-∞, 0)", 1.0),  # row 676
        ("numina_cn_k12", "(-∞, 0)", "(-∞, 0]", 0.0),  # row 676
        # percentages
        ("numina_olympiads", "83\\%", "83\\%", 1.0),  # row 3
        ("numina_olympiads", "83\\%", "83", 1.0),  # row 3
        ("numina_olympiads", "83\\%", "0.83", 1.0),  # row 3
        ("numina_olympiads", "83\\%", "84\\%", 0.0),  # row 3
        ("numina_cn_k12", "3.2\\%", "0.032", 1.0),  # row 946
        # fractions
        ("numina_synthetic_math", "\\frac{1}{4}", "\\dfrac{1}{4}", 1.0),  # row 5
        ("numina_synthetic_math", "\\frac{1}{4}", "1/4", 1.0),  # row 5
        ("numina_synthetic_math", "\\frac{1}{4}", "0.25", 1.0),  # row 5
        ("numina_synthetic_math", "\\frac{1}{4}", "\\frac{1}{5}", 0.0),  # row 5
        ("numina_synthetic_math", "\\frac{1}{4}", "\\frac{4}{1}", 0.0),  # row 5
        # numbers
        ("numina_synthetic_math", "22500", "22500", 1.0),  # row 0
        ("numina_synthetic_math", "22500", "22,500", 1.0),  # row 0
        ("numina_synthetic_math", "22500", "22500.0", 1.0),  # row 0
        ("numina_synthetic_math", "9007199254740993", "9007199254740993.0", 1.0),  # synthetic
        ("numina_synthetic_math", "9007199254740993", "+9007199254740993", 1.0),  # synthetic
        ("numina_synthetic_math", "9007199254740993", "+9007199254740993.0", 1.0),  # synthetic
        ("numina_synthetic_math", "-9007199254740993", "-9007199254740993.0", 1.0),  # synthetic
    ],
)
def test_unchanged_grading(data_source, ground_truth, answer, expected):
    assert _score(data_source, answer, ground_truth) == expected


# dropping the \left / \right delimiters must keep commands such as \leftarrow and \rightarrow
@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("A \\rightarrow B", "A \\rightarrow B", True),
        ("A \\leftarrow B", "A \\rightarrow B", False),
        ("\\left(A \\rightarrow B\\right)", "(A \\rightarrow B)", True),
        ("\\left(A \\leftarrow B\\right)", "(A \\rightarrow B)", False),
        ("\\left(A \\rightarrow B\\right)", "(A \\leftarrow B)", False),
    ],
)
def test_math_equal_keeps_arrows(prediction, reference, expected):
    assert math_equal(prediction, reference) is expected
