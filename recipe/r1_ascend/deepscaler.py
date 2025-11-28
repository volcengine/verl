# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
    solution_str = solution_str.strip()
    pattern = re.compile(r"<think>.*</think>.*<answer>.*\\boxed\{.*\}.*</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, solution_str)
    score = 0.0
    if format_match:
        score += 0.33

    extract_output = extract_boxed_content(solution_str)
    if grade_answer(extract_output, ground_truth):
        score += 0.67

    return score
