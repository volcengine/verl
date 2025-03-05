import asyncio
import aiohttp
from concurrent.futures import ProcessPoolExecutor
import torch
import json
from typing import List, Tuple
from verl.utils.swedev_utils import *
from verl import DataProto

class SWEDevRewardManager:
    """
    Modified Reward Manager that uses API calls to compute rewards for multi-turn conversations
    """

    def __init__(self, tokenizer, num_examine, compute_score=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # number of batches to print to console

    async def fetch_reward(self, sid: torch.Tensor, session: aiohttp.ClientSession) -> float:
        """Fetch reward from API for a single instance"""
        try:
            payload = {"sid": sid.item()}
            async with session.post(get_api(type="reward"), json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    result = await response.json()
                    return float(calc_reward(result))
                else:
                    print(f"fetch_reward - API request failed with text {response.text} for sid: {sid}")
                    return 0.0
        except Exception as e:
            print(f"fetch_reward - Error fetching reward for sid {sid}: {e}")
            return 0.0

    async def compute_reward(self, sid: str) -> float:
        """Compute reward for a single instance"""
        async with aiohttp.ClientSession() as session:
            return await self.fetch_reward(sid, session)

    async def process_batch(self, sids, response_length, batch_size, full_response_length):
        tasks = [self.compute_reward(sid) for sid in sids]
        scores = await asyncio.gather(*tasks)
        reward_tensor = torch.zeros([batch_size, full_response_length], dtype=torch.float32)

        for idx, score in enumerate(scores):
            reward_tensor[idx, response_length[idx] - 1] = score
            
        return reward_tensor
    
    def __call__(self, data: DataProto) -> torch.Tensor:
        """
        Compute rewards for multi-turn conversations using API calls

        Args:
            data (DataProto): Contains:
                'input_ids': Full conversation trajectory
                'attention_mask': Attention mask for the full sequence
                'instance_ids': List of instance IDs for API calls (assumed to be in non_tensor_batch)

        Returns:
            torch.Tensor: Reward tensor with shape [batch_size, sequence_length] where rewards are placed
                         at the last position of each response range
        """
        # If rm_scores already exist, return them
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # Get conversation data
        input_ids = data.batch['input_ids']
        batch_size = input_ids.shape[0]
        full_response_length = data.batch['responses'].shape[-1]
        response_mask = data.batch['attention_mask'][:, -full_response_length:]
        response_length = response_mask.sum(-1).float()  # (batch_size,)
        # reward_tensor = torch.zeros([batch_size, full_response_length], dtype=torch.float32)
        sids = data.batch['sids']
        reward_tensor = asyncio.run(self.process_batch(sids, response_length, batch_size, full_response_length))
        return reward_tensor