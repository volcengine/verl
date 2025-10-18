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

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Any, Awaitable
import torch
from torch import Tensor
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@dataclass
class RewardResult:
    score: float
    details: Dict[str, Any]

@register("native_async")
class AsyncNativeRewardManager(AbstractRewardManager):
    """Asynchronous reward manager with parallel scoring capabilities.
    
    Attributes:
        tokenizer: Tokenizer used for decoding sequences.
        num_examine: Number of samples to print for examination.
        compute_score: Async function to compute the reward score.
        reward_fn_key: Key to identify the data source in non_tensor_batch.
        max_concurrency: Maximum number of concurrent scoring requests.
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int = 1,
        compute_score: Optional[callable] = None,
        reward_fn_key: str = "data_source",
        max_concurrency: int = 20
    ) -> None:
        """Initialize the AsyncRewardManager.
        
        Args:
            tokenizer: Tokenizer instance for decoding sequences.
            num_examine: Number of samples to print for debugging/examination.
            compute_score: Async scoring function. Must be awaitable.
            reward_fn_key: Key to identify data source in non_tensor_batch.
            max_concurrency: Max concurrent scoring requests.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or self.default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def default_compute_score(
        self,
        data_source: str,
        solution_str: str,
        ground_truth: Any,
        extra_info: Optional[Any] = None
    ) -> RewardResult:
        """Default async scoring function that can be overridden."""
        # Implement your default async scoring logic here
        return RewardResult(score=0.0, details={})

    async def _decode_sequences(
        self,
        prompt_ids: Tensor,
        response_ids: Tensor,
        attention_mask: Tensor
    ) -> tuple[str, str, int]:
        """Decode prompt and response sequences asynchronously."""
        prompt_length = prompt_ids.shape[-1]
        
        valid_prompt_length = attention_mask[:prompt_length].sum().item()
        valid_response_length = attention_mask[prompt_length:].sum().item()
        
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        valid_response_ids = response_ids[:valid_response_length]
        
        # Run decoding in a thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        prompt_str, response_str = await asyncio.gather(
            loop.run_in_executor(
                None, 
                lambda: self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            ),
            loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            )
        )
        
        return prompt_str, response_str, valid_response_length

    async def _process_single_item(
        self,
        data_item: Any,
        print_counts: Dict[str, int]
    ) -> tuple[float, Dict[str, Any], int]:
        """Process a single data item asynchronously."""
        async with self.semaphore:
            # Decode sequences
            prompt_str, response_str, valid_response_length = await self._decode_sequences(
                prompt_ids=data_item.batch["prompts"],
                response_ids=data_item.batch["responses"],
                attention_mask=data_item.batch["attention_mask"]
            )
            
            # Get reward model info
            reward_model_info = data_item.non_tensor_batch["reward_model"]
            if isinstance(reward_model_info, str):
                reward_model_info = eval(reward_model_info)
            
            # Compute score
            score_result = await self.compute_score(
                data_source=data_item.non_tensor_batch[self.reward_fn_key],
                solution_str=response_str,
                ground_truth=reward_model_info["ground_truth"],
                extra_info=data_item.non_tensor_batch.get("extra_info", None)
            )
            
            # Update print counts and log if needed
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            if data_source not in print_counts:
                print_counts[data_source] = 0

            if print_counts[data_source] < self.num_examine:
                print_counts[data_source] += 1
                await self._log_sample(
                    prompt_str=prompt_str,
                    response_str=response_str,
                    ground_truth=reward_model_info["ground_truth"],
                    score_result=score_result
                )
            
            return score_result.score, score_result.details, valid_response_length

    async def _log_sample(
        self,
        prompt_str: str,
        response_str: str,
        ground_truth: Any,
        score_result: RewardResult
    ) -> None:
        """Log sample information asynchronously."""
        print(f"[prompt] {prompt_str}")
        print(f"[response] {response_str}")
        print(f"[ground_truth] {ground_truth}")
        
        if hasattr(score_result, 'details') and score_result.details:
            for key, value in score_result.details.items():
                print(f"[{key}] {value}")
        else:
            print(f"[score] {score_result.score}")

    async def async_compute_rewards(
        self, 
        data: DataProto
    ) -> tuple[Tensor, Dict[str, List]]:
        """Asynchronously compute rewards for all items in the batch."""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        print_counts = {}
        
        # Create all tasks
        tasks = [
            self._process_single_item(data[i], print_counts)
            for i in range(len(data))
        ]
        
        # Process tasks with limited concurrency
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        for i, (score, details, valid_length) in enumerate(results):
            reward_tensor[i, valid_length - 1] = score
            for key, value in details.items():
                reward_extra_info[key].append(value)
        
        return reward_tensor, dict(reward_extra_info)

    def __call__(
        self, 
        data: DataProto, 
        return_dict: bool = False
    ) -> Union[Tensor, Dict[str, Union[Tensor, Dict[str, List]]]]:
        """Synchronous interface that runs the async computation."""
        # Return pre-computed scores if available
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]
        
        # Run async computation
        loop = asyncio.get_event_loop()
        reward_tensor, reward_extra_info = loop.run_until_complete(
            self.async_compute_rewards(data)
        )
        
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        } if return_dict else reward_tensor