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

def compute_score_rule(
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
        "score": reward, "acc": acc, "pred": pred,
    }


FAPO_GENRM_TEMPLATE = (
    "The following is a math problem with its ground truth answer, along with an AI solution (split into steps):\n\n"
    "[Math Problem]\n\n"
    "{problem}\n\n"
    "[Ground Truth]\n\n"
    "{ground_truth}\n\n"
    "[AI Solution]\n\n"
    "{solution}\n\n"
    "Your task is to review and critique the solution step by step. Once you identify an error in a step, return the index of the step where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes 'not found').\n\n"
    "Please reason step by step, put your final answer (i.e., the index) in \\boxed{{}}."
)

async def compute_score_fapo(data_source, solution_str, ground_truth, extra_info, reward_model):
    question, split = extra_info["question"], extra_info["split"]
    solution_str = solution_str[-300:]
    correct, pred = verify(solution_str, ground_truth)

    grm_prompt = FAPO_GENRM_TEMPLATE.format(
        problem=question,
        ground_truth=ground_truth,
        solution=solution_str,
    )
    grm_prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": grm_prompt}],
        tokenize=True, add_generation_prompt=True,
    )
    grm_response = await reward_model.generate.remote(prompt_ids=grm_prompt_ids)
    breakpoint()

    # if not correct:
    #     return {"score": -1.0, "acc": correct, "pred": pred}
    # else:
    #     grm_prompt = FAPO_GENRM_TEMPLATE.format(
    #         problem=question,
    #         ground_truth=ground_truth,
    #         solution=solution_str,
    #     )
    #     grm_response = await reward_model.generate(grm_prompt)
    
