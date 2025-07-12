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
Custom reward model for DeepEyes, implementing an 'LLM-as-a-Judge' pattern.
Copy and modify from https://github.com/Visual-Agent/DeepEyes/blob/main/verl/utils/reward_score/vl_agent.py

This script defines a `compute_score` function that can be dynamically loaded by the
verl training framework. It evaluates a model's generated answer by calling an
external language model (the "Judge") and assigns rewards based on both the
correctness of the final answer and the effective use of tools.
"""

import os
import random
import re

import requests
from math_verify import parse, verify
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base_list = [
    # "http://172.30.52.123:8000/v1",
    # "http://10.39.3.123:18901/v1",
    os.environ.get("LLM_AS_A_JUDGE_BASE", "http://localhost:18901/v1"),
]

client_list = []
for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)
model_name_list = []
for client in client_list:
    response = requests.get(f"{api_base}/models")
    models = response.json()
    model_name_list.append(models["data"][0]["id"])


def get_chat_template():
    chat_template = (
        "Below are two answers to a question. Question is [Question], [Standard Answer] is the standard "
        "answer to the question, and [Model_answer] is the answer extracted from a model's output to "
        "this question.  Determine whether these two answers are consistent.\n"
        "Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. "
        "If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and "
        "'it is pink'.\n"
        "If they are consistent, Judgement is 1; if they are different, Judgement is 0. "
        "Just output Judgement and don't output anything else.\n"
    )
    return chat_template


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""  # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
"""  # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
"""  # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
"""  # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
"""  # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""  # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


COMMON_VERIFY_PROMPT = (
    "# CONTEXT #\n"
    "I am a teacher, and I have some high-level reasoning problems. I am tasked with evaluating the correctness of a "
    "student's answer. \n"
    "Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My "
    "job is to assess whether the student's answer captures the same meaning as the reference answer, even when "
    "expressed with different wording or format.\n\n"
    "# Overall Task #\n"
    "Based on the problem and the reference answer, my goal is to determine if the student's final answer is "
    "semantically equivalent to the reference answer. The judgement should be TRUE if the meaning of student's final "
    "answer is equivalent to the reference answer, even when expressed with different wording or format.\n\n"
    "# Rules for Judgement #\n"
    "- The judgement must be either TRUE or FALSE.\n"
    "- The judgement is TRUE if the student's final answer has the same meaning as the reference answer.\n"
    "- The judgement is FALSE if the student's final answer has a different meaning from the reference answer.\n"
    "- The judgement is FALSE even if the student's final answer almost correct with a minor mistakes.\n\n"
    "# Output Format #\n"
    "My output should be a single word: TRUE or FALSE. I must not include any other information in my response.\n"
    "## Equivalence Judgement"
)


MATH_VERIFY_PROMPT = (
    "# CONTEXT #\n"
    "I am a teacher, and I have some math problems. I am tasked with evaluating the correctness of a student's "
    "answer. \n"
    "Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My "
    "job is to assess whether the student's answer captures the same meaning as the reference answer, even when "
    "expressed with different wording or format.\n\n"
    "# Overall Task #\n"
    "Based on the problem and the reference answer, my goal is to determine if the student's final answer is "
    "mathematically equivalent to the reference answer. I need to focus on comparing the final numerical result or "
    "mathematical expression in both answers. Confirm that the mathematical relationships, variables, and "
    "operations conveyed are equivalent.\n\n"
    "# Rules for Judgement #\n"
    "- The judgement must be either TRUE or FALSE.\n"
    "- The judgement is TRUE if the student's final answer has the same meaning as the reference answer.\n"
    "- The judgement is FALSE if the student's final answer has a different meaning from the reference answer.\n"
    "- The judgement is FALSE even if the student's final answer almost correct with a minor mistakes.\n\n"
    "# Output Format #\n"
    "My output should be a single word: TRUE or FALSE. I must not include any other information in my response.\n"
    "## Equivalence Judgement"
)


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"

    return full_prompt


def extract_answer(text):
    """
    从给定的文本中提取<answer></answer>标签内部的内容。

    参数:
        text (str): 包含<answer>标签的文本

    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = solution_str.count("<think>")
    count_think_2 = solution_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = solution_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = solution_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = solution_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = solution_str.split("<answer>")[-1].split("</answer>")[0].strip()

    # pattern = re.compile(r'<\|im_start\|>assistant(.*?)$', re.DOTALL)  # 匹配最后一个 target 后的所有内容
    # match = pattern.search(predict_str)
    # if match:
    #     answer_text = match.group(1).strip()
    #     print(f'DEBUG{answer_text=}')
    # else:
    #     answer_text = ""

    question_text = extra_info["question"]
    full_prompt = get_prompt(answer_text, ground_truth, question_text)

    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        seed=random.randint(0, 1000000),
        temperature=0.3,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    response = chat_response.choices[0].message.content.strip()
    # print(response)
    if "Judgement:" in response:
        response = response.split("Judgement:")[-1].strip()
        if "1" in response:
            acc_reward = 1.0
        elif "0" in response:
            acc_reward = 0.0
        else:
            print(f" [WARNING] resp format error {response=}")
            acc_reward = 0.0
    else:
        if response == "1":
            acc_reward = 1.0
        elif response == "0":
            acc_reward = 0.0
        else:
            print(f" [WARNING] resp format error {response=}")
            acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    # tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    # reward 1
    # return 0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base
    # reward 2
    return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward

    # reward 2
    # return 1.0 * acc_reward + 0.2 * format_reward + 1.0 * tool_reward + 0.2 * tool_reward_base
    # reward 3
    # tool_reward_alpha = 1.2 if count_vision_1 > 0 else 0.0
    # return 1.0 * acc_reward * tool_reward_alpha + 0.2 * format_reward
    # reward 4
    # extra_reward = tool_reward_base * (count_vision_1 - 1) * (1 - acc_reward)
    # return  0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base  + 0.2 * extra_reward


def compute_common_reasoning(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = extract_answer(
        predict_no_think
    )  # predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
    if not answer_text:
        acc_reward = 0.0
        is_format_error = True
    elif len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True
    else:
        question_text = extra_info["question"]
        client_idx = random.randint(0, len(client_list) - 1)
        client = client_list[client_idx]
        model_name = model_name_list[client_idx]
        full_prompt = COMMON_VERIFY_PROMPT.format(
            query=question_text,
            gold_ans=ground_truth,
            pred_ans=answer_text,
        )

        acc_reward = 0.0
        for ix in range(8):
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed=random.randint(0, 1000000),
                temperature=0.5,
            )
            response = chat_response.choices[0].message.content.strip()
            judgement = response.split("## Equivalence Judgement")[-1].lower()
            if "true" in judgement and "false" not in judgement:
                acc_reward = 1.0
                break
            elif "false" in judgement and "true" not in judgement:
                acc_reward = 0.0
                break
            else:
                print(f" [ERROR] judgement format invalid: {judgement}")
                continue

    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    print(f" [DEBUG] query={extra_info['question']}, {ground_truth=}, {answer_text=}, {acc_reward=}, {format_reward=}")
    return 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward


def rule_math_verify(ground_truth, model_answer):
    gold = parse(ground_truth)
    answer = parse(model_answer)
    return verify(gold, answer)


def generative_verify(query, ground_truth, model_answer):
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    response = ""
    for it in range(8):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed=random.randint(0, 1000000),
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f" [ERROR math] generative_verify error: {e}")
            continue

    judgement = response.split("## Equivalence Judgement")[-1].lower()
    if "true" in judgement and "false" not in judgement:
        return True
    elif "false" in judgement and "true" not in judgement:
        return False
    else:
        print(" [ERROR math] verify bug output: ")


def compute_score_math(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    model_answer = ""
    predict_no_think = predict_str.split("</think>")[-1].strip()
    answer_pattern = r"\\boxed{([^}]+)}"
    answer_list = re.findall(answer_pattern, predict_no_think, flags=re.DOTALL)
    if len(answer_list) == 0:
        acc_reward = 0.0
        is_format_error = True
    else:
        if len(answer_list) > 1:
            is_format_error = True

        model_answer = answer_list[-1]
        if rule_math_verify(ground_truth, model_answer):
            acc_reward = 1.0
        else:
            acc_reward = 1.0 if generative_verify(extra_info["question"], ground_truth, model_answer) else 0.0

    format_reward = -1.0 if is_format_error else 0.0
    print(f" [DEBUG] query={extra_info['question']}, {ground_truth=}, {model_answer=}, {acc_reward=}, {format_reward=}")
    return 1.2 * acc_reward + 0.4 * format_reward


if __name__ == "__main__":
    predict_str = "The answer is 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    ground_truth = "left"
    extra_info = {
        "answer": "The woman is to the left of the man who is holding the camera.",
        "id": 0,
        "image": "/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg",
        "pred_ans": "The woman is to the right of the man who is holding the camera.",
        "question": "Is the woman to the left or to the right of the man who is holding the camera?",
    }

    score = compute_score("common_reasoning", predict_str, ground_truth, extra_info)
    print(f"Score: {score}")
