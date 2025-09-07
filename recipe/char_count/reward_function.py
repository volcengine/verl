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

"""
Reward function
"""

from verl.utils.reward_score import math_reward


def char_count_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    try:
        last_boxed_string = math_reward.last_boxed_only_string(solution_str)
        if last_boxed_string is None:
            return 0
        solution = math_reward.remove_boxed(last_boxed_string)
        if solution == ground_truth:
            return 1
        else:
            return 0
    except Exception:
        print(ground_truth, solution_str)
        return 0
