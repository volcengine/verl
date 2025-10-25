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

import json

import aiohttp

from verl.utils.reward_score.math_dapo import last_boxed_only_string, normalize_final_answer, remove_boxed


def verify(
    solution_str: str,
    gt: str,
) -> tuple[bool, str]:
    boxed_answer = last_boxed_only_string(solution_str)
    if boxed_answer is not None:
        extracted_answer = remove_boxed(boxed_answer)
    else:
        extracted_answer = "[INVALID]"

    pred = normalize_final_answer(extracted_answer)
    gt = normalize_final_answer(gt)
    return (pred == gt), pred


def compute_score_baseline(
    solution_str: str,
    ground_truth: str,
    **kwargs,
) -> float:
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }


ADDRESS = "xx.xx.xx.xx:xxxx"
MODEL_NAME = "FAPO-4B-GenRM"
FAPO_GENRM_TEMPLATE = (
    "The following is a math problem with its ground truth answer, along with an AI solution (split into steps):\n\n"
    "[Math Problem]\n\n"
    "{problem}\n\n"
    "[Ground Truth]\n\n"
    "{ground_truth}\n\n"
    "[AI Solution]\n\n"
    "{solution}\n\n"
    "Your task is to review and critique the solution step by step. "
    "Once you identify an error in a step, return the index of the step where the earliest error occurs. "
    "Otherwise, return the index of -1 (which typically denotes 'not found').\n\n"
    "Please reason step by step, put your final answer (i.e., the index) in \\boxed{{}}."
)


async def chat_completions_aiohttp(address, **chat_complete_request):
    try:
        request_url = f"http://{address}/v1/chat/completions"
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(
            url=request_url,
            json=chat_complete_request,
        ) as resp:
            output = await resp.text()
            try:
                output = json.loads(output)
                return output["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Error: {e}. Output: {output}")
                return ""
    finally:
        await session.close()


def judge_fp_process(response, return_err_step=False):
    try:
        boxed_result = last_boxed_only_string(response)
        result = remove_boxed(boxed_result)
        reward_score = int(eval(result)) != -1
        if return_err_step:
            return reward_score, int(result)
        return reward_score
    except Exception:
        if return_err_step:
            return None, None
        return None


async def compute_score_fapo(data_source, solution_str, ground_truth, extra_info, keep_genrm_critics=False, **kwargs):
    question, split = extra_info["question"], extra_info["split"]
    result = compute_score_baseline(solution_str, ground_truth)
    result["flawed_positive"] = False

    if split == "test" or result["acc"] == 0:
        if keep_genrm_critics:
            result["genrm_critics"] = ""
        return result
    else:
        prompt = FAPO_GENRM_TEMPLATE.format(problem=question, ground_truth=ground_truth, solution=solution_str)
        messages = [{"role": "user", "content": prompt}]
        response = await chat_completions_aiohttp(
            ADDRESS,
            messages=messages,
            model=MODEL_NAME,
            max_tokens=16384,
        )
        if response is not None and judge_fp_process(response):  # flawed positive
            result["score"] = 0.0
            result["flawed_positive"] = True

        if keep_genrm_critics and response is not None:
            result["genrm_critics"] = response

    return result
