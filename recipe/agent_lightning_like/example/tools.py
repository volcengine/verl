# Copyright 2025 Individual Contributor: linxxx3 (linxxx3@gmail.com)
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

from typing import Any

from agents import RunContextWrapper, function_tool

from verl.utils.reward_score.gsm8k import compute_score

from .apis import UserContext


def custom_err_handler(context: RunContextWrapper[Any], error: Exception) -> str:
    """A custom error handler to provide a user-friendly error message."""
    print(f"Error occurred during tool execution: {error}, context: {context}")
    return f"Error: {str(error)}"


# A demo Gsm8kTool for openai agent-sdk.
@function_tool(failure_error_function=custom_err_handler)
def calc_gsm8k_reward(wrapper: RunContextWrapper[UserContext], answer: str) -> str:
    """A tool for calculating the reward of gsm8k

    Args:
        answer: the answer to the question
    """
    ground_truth = wrapper.context.ground_truth
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)
    # "compute_score" expects answer is a string starts with "#### ",
    # but usually the tool call argument "answer" is a pure number
    if not isinstance(answer, str):
        answer = str(answer)
    if not answer.startswith("#### "):
        answer = "#### " + answer
    reward = compute_score(answer, ground_truth)
    return f"Current parsed {answer=} {reward=}"
