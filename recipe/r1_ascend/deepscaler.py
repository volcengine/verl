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

import re

from mathruler.grader import extract_boxed_content, grade_answer


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pattern = re.compile(r"<think>.*</think>.*<answer>.*\\boxed\{.*\}.*</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, solution_str)

    extract_output = extract_boxed_content(solution_str)
    if grade_answer(extract_output, ground_truth) and format_match:
        return 1.0
    else:
        return 0.0
