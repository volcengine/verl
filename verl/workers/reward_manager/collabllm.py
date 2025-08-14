
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, Optional, Union

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
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source"
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
    
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)

        data_source = data.non_tensor_batch["data_source"]
        messsages = data.non_tensor_batch["messages"]
        ground_truth = data.non_tensor_batch["ground_truth"]
        extra_info = data.non_tensor_batch["extra_info"]

        scores = self.compute_score(data_source, messsages, ground_truth, extra_info)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor