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

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

GENRM_PROMPT_TEMPLATE = """
The following is a math problem and an AI solution:

[Math Problem]

{problem}

[AI Solution]

{solution}

Your task is to review and critique the solution step by step, and output whether the AI solution is correct.

Please put your final answer (i.e., 'True' or 'False') in \\boxed{{}}.
""".strip()


def construct_genrm_inputs_from_rollouts(rollout_question, rollout_response, ground_truth=None) -> str:
    prompt = GENRM_PROMPT_TEMPLATE.format(problem=rollout_question, solution=rollout_response)
    return prompt


def _compute_reward_score(response: str) -> float:
    reward_score = 0.0
    try:
        boxed_result = last_boxed_only_string(response)
        if boxed_result is not None:
            result = remove_boxed(boxed_result)
            reward_score = float(result == "True")
    except Exception as e:
        print(e)
    return reward_score


def convert_genrm_responses_to_rewards(output: str) -> float:
    return _compute_reward_score(output)
