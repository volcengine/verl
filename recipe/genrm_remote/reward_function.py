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

import re
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import requests

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

BASE_URL = "http://0.0.0.0:8000"
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_WORKERS = 32
MODEL_NAME = "genrm-demo"
GENRM_PROMPT_TEMPLATE = """
Your task is to evaluate whether the answer is good based on the following criteria:

Quality Criteria:
1. RELEVANCE: The answer directly addresses what the query is asking for
2. SPECIFICITY: The answer is specific and concrete, not generic or vague
3. CONCISENESS: The answer is brief and to the point (typically 1-5 words)
4. ACCURACY: For factual queries, the answer should be correct
5. APPROPRIATENESS: The answer matches the query type and expected response format

Query Type Guidelines:
- FACTUAL QUERIES: Should provide the direct fact/answer 
  (e.g., "Paris" for "capital of France")
- RECOMMENDATION QUERIES: Should give popular/well-known options 
  (e.g., "Python" for "best first programming language")
- LOCATION QUERIES: Should provide specific place names 
  (e.g., "Tokyo" for "good place to visit in Japan")
- DEFINITION QUERIES: Should provide the specific term or concept 
  (e.g., "Avocado" for "main ingredient in guacamole")

Good Answer Examples:
- Query: "What is the capital of France?" → Answer: "Paris" ✓
- Query: "What is a famous travel destination in France?" → Answer: "Paris" ✓
- Query: "What programming language is good for beginners?" → Answer: "Python" ✓
- Query: "What is the largest ocean?" → Answer: "Pacific Ocean" ✓

Bad Answer Examples:
- Query: "What is the capital of France?" → Answer: "The capital city of France is a 
  beautiful place called Paris" ✗ (too verbose)
- Query: "What is a famous travel destination in France?" → Answer: "There are many 
  places" ✗ (too vague)
- Query: "What programming language is good for beginners?" → Answer: "Programming 
  languages" ✗ (not specific)

Please put your final evaluation (i.e., 'True' for good answer or 'False' for bad answer) 
in \\boxed{{}}.
""".strip()


def get_response(problem, solution_str):
    prompt = GENRM_PROMPT_TEMPLATE.format(problem=problem, solution=solution_str)
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            headers = {"Content-Type": "application/json"}
            chat_url = f"{BASE_URL}/v1/chat/completions"
            data = {"model": MODEL_NAME, "messages": messages}
            output = requests.post(chat_url, headers=headers, json=data, timeout=30)
            response = output.json()["choices"][0]["message"]["content"]
            return response
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print("Exception: ", repr(e))
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                sleep(delay)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")

    raise ConnectionRefusedError(f"Failed to run the model for {prompt}!")


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


def compute_score(data_source, solution_str, extra_info):
    split = extra_info["split"]

    # todo -- add real search here to get the results, then we can calculate relevance.
    solution_str = (re.search(r"####\s*(.*)", solution_str) or [None, None])[1]

    if split == "test":
        problem = extra_info["question"]
        response = get_response(problem, solution_str)
        if response is not None:
            reward_score = compute_reward(response)
        else:
            reward_score = 0.0

        return reward_score
    else:
        problem = extra_info["question"]
        response = get_response(problem, solution_str)
        if response is not None:
            reward_score = compute_reward(response)
        else:
            reward_score = 0.0

        return reward_score


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for data_source, solution_str, extra_info in zip(
            data_sources, solution_strs, extra_infos, strict=True
        ):
            future = executor.submit(compute_score, data_source, solution_str, extra_info)
            futures.append(future)

        results = [future.result() for future in futures]

    return results
