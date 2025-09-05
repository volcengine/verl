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

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple
import asyncio

import torch
import json
import ast
from numpy import array  

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.format import validate_hybird_format

from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import ipdb 
import re
from openai import OpenAI
from openai import AsyncOpenAI

import concurrent.futures
from functools import partial
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


PROMPT_TEMPLATE = """###角色
   @@ground_truth是@@question的正确答案；@@answer是智能问答机器人回答的答案
   请你充当公正的裁判，对@@answer评分，分值必须大于等于0小于等于3分，得分不能为负数，如果出现负数，需要按照规则重新评分。
   
###评分规则
      1、准确性：
          ground_truth和answer内容完全一致，必须评为3分，如ground_truth为“1”，answer为“1”，评分必须为3分。
          答案必须正确反映ground_truth中的信息，任何与正确答案冲突或矛盾的信息，或不包含在正确答案中的信息，都会导致答案被评为0分。
         如果答案中包含任何错误信息，即使只有一小部分，也应根据规则被评为0分。
      2、完整性：答案需要包含所有关键信息。缺少关键信息会根据缺失内容的数量和重要性扣分。
      3、相关性：答案应与问题直接相关，不包含不相关信息。包含少量不相关信息可能会扣分，而大量不相关信息可能导致答案被评为0分。

        
###评分梯度
      0分：内容完全错误、严重缺失或与问题无关；无逻辑；或明确表示信息不可用。或直接回答“在提供的【参考文本】中，并没有直接提及...”
      1分：包含部分正确答案的核心内容，但关键信息缺失或错误，或包含大量不相关信息。
      2分：内容基本准确，包含大部分问题核心内容，但不全面，缺少一部分关键信息或包含少量不相关信息。
      3分：内容完整，准确，覆盖所有问题核心内容，无错误信息，无额外不相关信息，条理清晰。
         
###输出规则
   请严格遵循以下示例的格式输出评分结果："[[rating]]", 示例: "Rating: [[0]]"。
   

[Question]
%s

[The Start of Answer]
%s
[The End of Answer]

[The Start of Ground Truth]
%s
[The End of Ground Truth] 
/no_think
"""

@dataclass
class ScoreInput:
    question: str
    response_str: str
    ground_truth: str

@dataclass
class ScoreOutput:
    score: float
    detail: str

def build_prompt(question: str, answer: str, ground_truth: str) -> str:
    """Build complete prompt from template"""
    return PROMPT_TEMPLATE % (question, answer, ground_truth)


class AsyncQwenScorer:
    """Async scorer using OpenAI client"""
    
    def __init__(self, base_url="http://10.55.42.83:31691/v1", api_key="EMPTY"):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = "Qwen3-32B_for_kefu"
        

    async def async_compute_score(self, question: str, response_str: str, ground_truth: str) -> Tuple[float, str]:
        """Compute score asynchronously"""
        prompt = build_prompt(question, response_str, ground_truth)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
    
            content = response.choices[0].message.content
            match = re.search(r'\[\[(\d+)\]\]', content)

            format_score = validate_hybird_format(response_str, question)

            if match:
                score = float(match.group(1))
                if 0 <= score <= 3:
                    # return (score / 3) * 0.75 + format_score * 0.25, content
                    return (score / 3), content
                else:
                    return -1, f"评分超出范围: {content}"
            return 0.0, "No score found in response"

        except Exception as e:
            return 0.0, f"Error in scoring: {str(e)}"

@register("intelligent_kefu_async")
class AsyncKefuRewardManager(AbstractRewardManager):
    """Async reward manager with concurrent scoring"""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.scorer = AsyncQwenScorer()

    async def _process_item(self, data_item) -> Tuple[ScoreInput, int]:
        """Extract scoring inputs from data item"""
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        
        # Get valid prompt and response lengths
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        if '</think>' in response_str:
            parts = response_str.split('</think>')
            if len(parts) > 1:
                answer_part = parts[1].strip()
            else:
                answer_part = response_str
        else:
            answer_part = response_str
      #   print(answer_part)

        question = prompt_str.split('\n')[1].split('\nassistant')[0].strip()
        
        if isinstance(data_item.non_tensor_batch["reward_model"], str):
            data_item.non_tensor_batch["reward_model"] = eval(
                data_item.non_tensor_batch["reward_model"]
            )
        
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        
        return ScoreInput(
            question=question,
            response_str=answer_part,
            ground_truth=ground_truth
        ), valid_response_length

    async def batch_compute_scores(self, inputs: List[ScoreInput], max_concurrency: int = 40) -> List[ScoreOutput]:
        """Process batch of inputs with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def bounded_score(input_item: ScoreInput):
            async with semaphore:
                score, detail = await self.scorer.async_compute_score(
                    input_item.question,
                    input_item.response_str,
                    input_item.ground_truth
                )
                return ScoreOutput(score=score, detail=detail)
        
        return await asyncio.gather(*[bounded_score(item) for item in inputs])

    async def async_compute_rewards(self, data: DataProto) -> dict:
        """Async version of reward computation"""
        if "rm_scores" in data.batch.keys():
            return {"reward_tensor": data.batch["rm_scores"]}
        
        # Prepare all inputs
        inputs = []
        valid_lengths = []
        for i in range(len(data)):
            input_item, valid_len = await self._process_item(data[i])
            inputs.append(input_item)
            valid_lengths.append(valid_len)
        
        results = await self.batch_compute_scores(inputs)
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_details = []
        
        for i, (result, valid_len) in enumerate(zip(results, valid_lengths)):
            reward_tensor[i, valid_len - 1] = result.score
            reward_details.append(result.detail)
            
            if i < self.num_examine:
                print(f"Sample {i}:\nQuestion: {inputs[i].question}\n"
                      f"Response: {inputs[i].response_str}\n"
                      f"Score: {result.score}\n")
        
        return {
            "reward_tensor": reward_tensor,
            "reward_details": reward_details
        }

    def __call__(self, data: DataProto, return_dict=False):
        """Synchronous interface that runs async code"""
        results = asyncio.run(self.async_compute_rewards(data))
        
        if return_dict:
            return results
        return results["reward_tensor"]
