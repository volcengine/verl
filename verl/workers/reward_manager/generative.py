import asyncio
from openai import AsyncOpenAI
import numpy as np
import re
import torch
from typing import List, Dict, Any
from verl.utils.reward_score.generative import process_data_async
from transformers import AutoTokenizer


def _compute_score(data_source: List[str], solution_str: List[str], ground_truth: List[str],
                   extra_info: List[Dict[str, Any]], config: Dict[str, Any]) -> torch.Tensor:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(process_data_async(data_source, solution_str, ground_truth, extra_info, config))


class GenerativeRewardManager:

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        config: Dict[str, Any] = None,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = _compute_score if compute_score is None else compute_score
        self.config = config

    def __call__(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        # Extract necessary information
        data_sources = []
        solution_strs = []
        ground_truths = []
        extra_infos = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode sequences
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            # Extract other information
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # Append to lists
            data_sources.append(data_source)
            solution_strs.append(sequences_str)
            ground_truths.append(ground_truth)
            extra_infos.append(extra_info)

            # Print examples for debugging
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # Batch compute scores
        scores = self.compute_score(
            data_source=data_sources,
            solution_str=solution_strs,
            ground_truth=ground_truths,
            extra_info=extra_infos,
            config=self.config,
        )

        # Fill scores into reward_tensor
        for i, score in enumerate(scores):
            data_item = data[i]
            valid_response_length = data_item.batch['attention_mask'][len(data_item.batch['prompts']):].sum()
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor
