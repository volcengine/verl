import copy
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Tuple

from tqdm import tqdm

import re
import asyncio
import aiohttp

import random
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed



prompt_llm_score = """You are a strict and careful answer grader. Your job is to determine whether the student's FINAL answer is logically or mathematically correct
when compared to the standard answer.

## Input

【Question】
{query}

【Standard Answer】
{ground_truth}

【Student Answer】
{response}

## Task

1. **Interruption Check**
   - Determine whether the student's answer is cut off or interrupted.
   - If the answer is interrupted, ignore the following rules and output /boxed{INTERRUPT} at the end.

2. **Question Understanding**
   - Read the question carefully. Determine whether:
     - the answer order matters,
     - multiple answers are required,
     - only one possible solution is sufficient,
     - the question has subquestions.
   - These constraints MUST be respected when grading.

3. **Final Answer Identification**
   - Ignore all intermediate answers or reasoning.
   - ONLY consider the LAST answer given by the student, even if it is a guess.

4. **Final Answer Extraction**
   - Clearly identify what the student’s final answer is.
   - Re-state the student’s final answer in your own words.
   - Ignore any parts of the question text repeated in the student answer.
   - If the question has multiple subquestions, collect the final answer for EACH required subquestion.

5. **Format Normalization**
   - If the student’s answer format differs from the standard answer:
     - Convert the student’s answer into the standard answer’s format before comparison.

6. **Final Correctness Judgment**
   - After normalization, determine whether the student’s final answer is:
     - logically correct, OR
     - mathematically equivalent.
   - Unnecessarily complicated answers are acceptable if they are correct.
   - If correct, output /boxed{CORRECT}.
   - Otherwise, output /boxed{WRONG}.

## Important Notes
- Spelling errors are WRONG.
- Wrong order is WRONG if order matters.
- Answering non-existent subquestions is WRONG.
- If interrupted, output `/boxed{INTERRUPT}` regardless of correctness.

## Output Rules
- Your final output MUST be exactly one of [/boxed{CORRECT}, /boxed{WRONG}, /boxed{INTERRUPT}]
"""




def vllm_request(url, model_name_or_path, query_list, system=None, n=1, t=0.7, max_tokens=2048, top_p=0.8, repetition_penalty=1.05, skip_special_tokens=True, sk="token-abc123", history: list=None):
    client = OpenAI(
        base_url=url,
        api_key=sk,
        timeout=600.0
    )
    if isinstance(query_list, str):
        query_list = [query_list]

    # check something: when n > 1, only single-turn requests are allowed.
    if n > 1 and len(query_list) > 1:
        print("[warning] Multi-turn requests will set `n` to 1.")
        n = 1
    assert n > 0, "`n` should be set to a positive integer."

    messages = []
    if system:
        messages = [{"role": "system", "content": system}]

    if (history is None or len(history) == 0):
        pass
    else:
        messages += copy.deepcopy(history)

    responses = []
    for prompt in query_list:
        messages.append({"role": "user", "content": prompt})
        completion = client.chat.completions.create(
            model=model_name_or_path,
            messages=messages,
            temperature=t,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            extra_body={
                "repetition_penalty": repetition_penalty,
                "skip_special_tokens": skip_special_tokens,
            },
        )
        if len(completion.choices) > 1:
            responses = [choice.message.content for choice in completion.choices]
            messages.append({"role": "assistant", "content": responses[0]})
            break
        else:
            response = completion.choices[0].message.content
            messages.append({"role": "assistant", "content": response})
            responses.append(response)
    
    return responses, messages




# {
#     "index": i,
#     "question": data.non_tensor_batch["question"],
#     "stu_answer": student_answer,
#     "stu_resp": resp_str,
#     "ground_truth": ground_truth
# }
def process_one(data_item, index, model_type="vllm"):
    urls = [
        "http://10.95.237.101:8000/v1",
        "http://10.95.237.101:8001/v1",
        "http://10.95.237.101:8002/v1",
        "http://10.95.237.101:8003/v1",
        "http://10.95.237.79:8000/v1",
        "http://10.95.237.79:8001/v1",
        "http://10.95.237.79:8002/v1",
        "http://10.95.237.79:8003/v1",
        "http://10.95.245.164:8000/v1",
        "http://10.95.245.164:8001/v1",
        "http://10.95.245.164:8002/v1",
        "http://10.95.245.164:8003/v1",
        "http://10.95.239.22:8000/v1",
        "http://10.95.239.22:8001/v1",
        "http://10.95.239.22:8002/v1",
        "http://10.95.239.22:8003/v1"
    ]
    
    model_id = "Qwen2.5-32B-Instruct"

    url = random.choice(urls)

    prompt = [prompt_llm_score.replace("{query}", data_item["question"]).replace("{ground_truth}", data_item["ground_truth"]).replace("{response}", data_item["stu_resp"])]


    if model_type == "vllm":
        responses, messages = vllm_request(
            url, model_id, prompt,
            system=None,
            sk="123", 
            n=1,
            max_tokens=2048*4, 
            t=0.9, 
            top_p=0.9,
            repetition_penalty=1.05,
            skip_special_tokens=0,
            history=[]
        )

    return index, responses
    
# 请求大模型判断答案是否正确
def compute_llm_scores(list_of_data, n_parallel=128):

    # raise RuntimeError("【调试】代码运行到了 compute_llm_scores 的调用阶段！")

    
    model_type = "vllm"

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {
            executor.submit(
                process_one,
                data_item,
                index, 
                model_type
            ): index
            for index, data_item in enumerate(list_of_data)
        }

        for future in as_completed(futures):
            index, processed = future.result()

            processed_str = processed[0]
            # 从返回值中提取结果
            match = re.search(r'/boxed\{(.*?)\}', processed_str)
            if match:
                answer = match.group(1).strip()
            else:
                answer = ""

            if answer.lower() == "correct":
                score = 1
            elif answer.lower() == "wrong":
                score = 0
            else:
                score = 0 # 提取失败 or Interupt
            
            list_of_data[index]["score"] = score

    return list_of_data
    