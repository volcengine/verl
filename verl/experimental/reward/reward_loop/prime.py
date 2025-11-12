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
import inspect
import logging
import os

from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score

logger = logging.getLogger(__file__)


@register("prime")
class PrimeRewardLoopManager(RewardLoopManagerBase):
    """
    Reward loop manager with rate limiting support for API-based reward functions.

    This manager is designed for LLM-as-judge and other external API-based reward functions
    that require rate limiting to prevent overwhelming the service or hitting API quotas.

    Key features:
    - Global rate limiting via class-level semaphore (shared across all agent loop workers)
    - Timeout protection (default 5 minutes per reward computation)
    - Graceful error handling (returns 0 score on failure instead of crashing)
    - Support for both async and sync reward functions
    - Compatible with tool_extra_fields from agent loop

    Configuration:
        reward_model:
            max_concurrent: 1  # Maximum concurrent reward function calls
            launch_reward_fn_async: True
        reward_manager: "prime"

    Recommended max_concurrent values:
    - 1: Low-tier APIs (gpt-5-nano 200K TPM, OpenAI Tier 1 with 500 RPM)
    - 2-4: OpenAI Tier 2 (5000 RPM, 2M+ TPM)
    - 8-16: OpenAI Tier 3 (10K+ RPM, 10M+ TPM)
    - 32-64: OpenAI Tier 4+ or self-hosted APIs
    """

    # Class-level semaphore shared across all instances for global rate limiting
    _semaphore = None
    _max_concurrent = None
    _prime_class_initialized = False

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances.

        This creates a class-level semaphore that is shared by all PrimeRewardLoopManager
        instances, ensuring true global rate limiting across all agent loop workers.
        """
        # Call parent init_class first
        super().init_class(config, tokenizer)

        # Use our own class-level flag to avoid conflicts with base class
        if cls._prime_class_initialized:
            return

        cls._max_concurrent = config.reward_model.get("max_concurrent", 1)
        cls._semaphore = asyncio.Semaphore(cls._max_concurrent)

        logger.info(
            f"[PrimeRewardLoopManager] Rate limiting enabled with max_concurrent={cls._max_concurrent}. "
            f"This semaphore is shared across all agent loop workers for global rate limiting."
        )

        cls._prime_class_initialized = True

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        # Timeout for reward computation (5 minutes default)
        # Can be overridden via config
        self.timeout = config.reward_model.get("timeout", 300.0)

    async def _compute_reward(
        self, data_source: str, solution_str: str, ground_truth: str, extra_info: dict
    ) -> dict | float:
        """Execute reward computation with proper async/sync handling.

        Args:
            data_source: Source identifier for the data
            solution_str: Generated solution string
            ground_truth: Ground truth answer
            extra_info: Additional information including tool fields

        Returns:
            Reward score as float or dict with 'score' key
        """
        if self.is_async_reward_score:
            # Async reward function - call directly
            return await self.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                reward_router_address=self.reward_router_address,
                reward_model_tokenizer=self.reward_model_tokenizer,
            )
        else:
            # Sync reward function - run in executor
            return await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    reward_router_address=self.reward_router_address,
                    reward_model_tokenizer=self.reward_model_tokenizer,
                ),
            )

    async def run_single(self, data: DataProto) -> dict:
        """Compute reward for a single data item with rate limiting.

        This method:
        1. Extracts and decodes the response
        2. Acquires semaphore (blocks if max_concurrent limit reached)
        3. Calls reward function with timeout protection
        4. Releases semaphore
        5. Returns formatted reward result

        Args:
            data: DataProto containing a single data item

        Returns:
            dict with keys:
                - reward_score: float, the computed reward (0.0 on error/timeout)
                - reward_extra_info: dict with additional info (acc, timeout, error, etc.)
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        # Extract response
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Extract metadata
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        # Merge tool_extra_fields if present (from agent loop with tools)
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        # Decode response
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        reward_extra_info = {}

        # Apply rate limiting and compute reward
        async with self._semaphore:
            try:
                # Compute reward with timeout protection
                result = await asyncio.wait_for(
                    self._compute_reward(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    ),
                    timeout=self.timeout,
                )

                # Parse result
                score: float
                if isinstance(result, dict):
                    score = result["score"]
                    # Copy all fields from result to reward_extra_info
                    for key, value in result.items():
                        reward_extra_info[key] = value
                else:
                    score = result
                    reward_extra_info["acc"] = score

                reward = score

            except asyncio.TimeoutError:
                logger.warning(
                    f"[Timeout] Reward computation timed out after {self.timeout}s for data_source={data_source}. "
                    f"Response preview: {response_str[:100]}..."
                )
                reward = 0.0
                reward_extra_info["timeout"] = True
                reward_extra_info["acc"] = 0.0

            except Exception as e:
                logger.error(
                    f"[Error] Reward computation failed for data_source={data_source}: {e}. "
                    f"Response preview: {response_str[:100]}..."
                )
                reward = 0.0
                reward_extra_info["error"] = str(e)
                reward_extra_info["acc"] = 0.0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
