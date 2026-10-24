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
"""CPU unit tests for prime_code call-based grading across APPS/TACO list and JSON-line layouts (verl#8174)."""

import json

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.prime_code.utils import check_correctness


def test_prime_code_call_based_apps_list_and_string_layouts():
    """Verify default_compute_score grades call-based solutions in both APPS/TACO list and JSON-line string layouts."""
    correct_solution = "```python\ndef add(a, b):\n    return a + b\n```"
    wrong_solution = "```python\ndef add(a, b):\n    return a - b\n```"

    apps_layout = {"fn_name": "add", "inputs": [[1, 2], [3, 4]], "outputs": [3, 7]}
    string_layout = {"fn_name": "add", "inputs": ["1\n2", "3\n4"], "outputs": ["3", "7"]}

    assert default_compute_score("apps", correct_solution, json.dumps(apps_layout)) == 1.0
    assert default_compute_score("apps", correct_solution, json.dumps(string_layout)) == 1.0
    assert default_compute_score("apps", wrong_solution, json.dumps(apps_layout)) == 0.0


def test_check_correctness_call_based_preserves_string_return_values_and_truncates_metadata():
    """Verify check_correctness handles native string expected outputs and wrong-answer metadata in list layout."""
    in_outs = {
        "fn_name": "greet",
        "inputs": [["world"], ["verl"]],
        "outputs": ["hello world", "hello verl"],
    }
    res, metadata_list = check_correctness(
        in_outs=in_outs,
        generation="def greet(name):\n    return 'hello ' + name\n",
        timeout=5,
        debug=False,
    )
    assert res == [True, True]
    assert list(metadata_list) == [{}]

    # Verify wrong answer metadata formatting when inputs/outputs are native objects
    in_outs_fail = {
        "fn_name": "greet",
        "inputs": [["world"]],
        "outputs": ["hello world"],
    }
    res_fail, meta_fail_list = check_correctness(
        in_outs=in_outs_fail,
        generation="def greet(name):\n    return 'bye ' + name\n",
        timeout=5,
        debug=False,
    )
    assert res_fail == [False]
    meta_fail = list(meta_fail_list)[0]
    assert meta_fail["error_message"] == "Wrong Answer"
    assert isinstance(meta_fail["inputs"], str)
    assert isinstance(meta_fail["expected"], str)

    # Verify native dictionary keys in list layout are preserved without int coercion
    in_outs_dict = {
        "fn_name": "lookup",
        "inputs": [[{"1": 5}, "1"]],
        "outputs": [5],
    }
    res_dict, _ = check_correctness(
        in_outs=in_outs_dict,
        generation="def lookup(mapping, key):\n    return mapping[key]\n",
        timeout=5,
        debug=False,
    )
    assert res_dict == [True]
