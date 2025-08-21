
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Optional, Union

import numpy as np
import copy
import psutil
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker


TERMINATION_SIGNAL = "[[TERMINATE CHAT]]"


@register("collabllm")
class CollabLLMRewardManager(AbstractRewardManager):
    """
    The Reward Manager used in https://github.com/Wuyxin/collabllm/
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        metric_weights: dict,
        llm_judge_kwargs: dict,
        reward_fn_key: str = "data_source",
        compute_score: Optional[Callable] = None,
        normalize_by_data_source = False
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

        self.metric_weights = metric_weights
        self.llm_judge_kwargs = llm_judge_kwargs
        self.normalize_by_data_source = normalize_by_data_source

        self.metrics = list(self.metric_weights.keys())
    
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # Use asyncio.run to handle the async computation
        return asyncio.run(self._compute_rewards_async(data, return_dict))
    
    async def _compute_rewards_async(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)

        data_source = data.non_tensor_batch["data_source"]
        ground_truth = data.non_tensor_batch["ground_truth"]
        extra_info = data.non_tensor_batch["extra_info"]
        message_lst = data.non_tensor_batch["messages"]

        # batch the messages into multiple 
        num_repeat_rollouts = len(message_lst[0]["messages"])
        batch_size = len(data_source)
        grouped_messages = [[message_lst[i]["messages"][j] for i in range(len(message_lst))] for j in range(num_repeat_rollouts)]

        # Flatten lists for all batch items across all rollouts
        flattened_data_sources = [data_source[i] for _ in range(num_repeat_rollouts) for i in range(batch_size)]
        flattened_ground_truths = [ground_truth[i] for _ in range(num_repeat_rollouts) for i in range(batch_size)]
        flattened_extra_infos = [extra_info[i] for _ in range(num_repeat_rollouts) for i in range(batch_size)]
        flattened_messages = [grouped_messages[j][i] for j in range(num_repeat_rollouts) for i in range(batch_size)]

        tasks = [
            self.compute_score(flattened_data_sources[i], flattened_messages[i], 
                             flattened_ground_truths[i], flattened_extra_infos[i], self.metrics, **self.llm_judge_kwargs)
            for i in range(len(flattened_data_sources))
        ]
        score_dicts = await asyncio.gather(*tasks)

        # Aggregate scores for each metric across repeated rollouts
        scores_by_metrics = {
            metric: torch.stack(
                [score_dict[metric] for score_dict in score_dicts]
            ).view(num_repeat_rollouts, -1).sum(dim=0)
            for metric in self.metrics
        }

        # Apply metric-specific weights
        weighted_scores_by_metrics = {
            metric: scores_by_metrics[metric] * self.metric_weights[metric]
            for metric in self.metrics
        }

        # Compute mean of weighted scores for each metric
        mean_weighted_scores_by_metrics = {
            metric: weighted_scores_by_metrics[metric].mean(dim=0)
            for metric in self.metrics
        }

        # Combine weighted scores from all metrics into a single tensor
        scores = torch.stack(
            [weighted_scores_by_metrics[metric] for metric in self.metrics]
        ).sum(dim=0)
        print('Scores:', scores, mean_weighted_scores_by_metrics)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[0].item() - 1] = scores[i]

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor