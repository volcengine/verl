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

import re
import asyncio
import numpy as np
import torch
from typing import List, Dict, Any
from openai import AsyncOpenAI
from transformers import AutoTokenizer


def extract_output(solution_text: str):
    # Match everything inside the last \boxed{} in the solution text
    boxed_pattern = r'\\bold{(.*)}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None


async def _query_openai_async(client: AsyncOpenAI, sequence_str: str, config) -> float:
    """
    Query OpenAI API asynchronously.
    """
    max_retries = config['max_retries']
    retry_count = 0
    scoring_prompt = config['scoring_prompt']
    while retry_count < max_retries:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": scoring_prompt + '\n' + sequence_str
            },
        ]
        # if using vllm server
        if config['tokenizer'] and config['apply_chat_template']:
            tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
            messages = tokenizer['apply_chat_template'](messages, tokenize=False, add_generation_prompt=True)
        try:
            response = await client.chat.completions.create(
                model=config['model_name'],
                messages=messages,
                max_tokens=config['max_tokens'],
                temperature=config['temperature'],
            )
            try:
                scores = response.choices[0].message.content.strip()
                score = float(extract_output(scores))
                return score
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    return 0
                continue  # Retry the request
        except Exception as e:
            print(f"Error querying OpenAI API: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                print("Max retries reached. Returning default score.")
                return config['default_score']
            continue  # Retry the request


async def process_data_async(data_source: List[str], solution_str: List[str], ground_truth: List[str],
                             extra_info: List[Dict[str, Any]], config) -> torch.Tensor:
    """
    Process data asynchronously using OpenAI API.
    """
    reward_tensor = torch.zeros(len(solution_str), dtype=torch.float32)
    client = AsyncOpenAI(api_key=config['api_key'], base_url=config['api_base'])

    tasks = []
    for i in range(len(solution_str)):
        prompt = solution_str[i]
        response = ground_truth[i]
        if response is None:
            sequence_str = prompt
        else:
            sequence_str = f"{prompt}\nReference:\n{response}"

        task = asyncio.create_task(_query_openai_async(client, sequence_str, config))
        tasks.append((i, task))

    # Execute all tasks concurrently
    for i, task in tasks:
        result = await task
        reward_tensor[i] = result

    return reward_tensor