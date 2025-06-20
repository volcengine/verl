# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio

from openai import AsyncOpenAI

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

GENRM_PROMPT_TEMPLATE = """
The following is a math problem and an AI solution:

[Math Problem]

{problem}

[AI Solution]

{solution}

Your task is to review and critique the solution step by step, and output whether the AI solution is correct.

Please put your final answer (i.e., 'True' or 'False') in \\boxed{{}}.
""".strip()

BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2


async def get_response(client, problem, solution_str):
    prompt = GENRM_PROMPT_TEMPLATE.format(problem=problem, solution=solution_str)
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            output = await client.chat.completions.create(
                model="genrm-demo",
                messages=messages,
                max_tokens=4096,
                temperature=0.0,
            )
            return output.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2**attempt)
                print(f"Connection error: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")
    return None


def compute_reward(response):
    reward_score = 0.0
    try:
        boxed_result = last_boxed_only_string(response)
        if boxed_result is not None:
            result = remove_boxed(boxed_result)
            reward_score = float(result == "True")
    except Exception as e:
        print(e)
    return reward_score


async def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    split = extra_info["split"]
    from verl.utils.reward_score.gsm8k import compute_score

    func_rm_score = compute_score(solution_str, ground_truth)
    if split == "train":
        return func_rm_score
    else:  # split = "train"
        if func_rm_score == 0.0:
            return func_rm_score

        client = AsyncOpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            timeout=90,
        )
        problem = extra_info["question"]
        response = await get_response(client, problem, solution_str)
        if response is not None:
            reward_score = compute_reward(response)
        else:
            reward_score = 0.0

        return reward_score


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    tasks = [compute_score(data_source, solution_str, ground_truth, extra_info) for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos)]
    results = loop.run_until_complete(asyncio.gather(*tasks))
    return results
