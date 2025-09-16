# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

from collections import defaultdict
from typing import Any
import torch
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.think_utils import extract_think_content
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

@register("think_aware")
class ThinkAwareRewardManager(AbstractRewardManager):
    """Think-aware reward manager that only computes rewards on answer parts"""
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.already_print_data_sources = {}
    
    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        for i in range(len(data)):
            data_item = data[i]
            
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # 解码
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # 提取answer部分（去除think内容）
            think_data = extract_think_content(response_str)
            answer_str = think_data["answer_content"]
            
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            # 只对answer部分计算奖励
            score = self.compute_score(
                data_source=data_source,
                solution_str=answer_str,  # 只使用answer部分
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            
            reward_tensor[i] = reward
            
            # Debug输出（保持与原有逻辑一致）
            if self.num_examine > 0 and data_source not in self.already_print_data_sources:
                print(f"=== Think-Aware Reward Debug ===")
                print(f"Data source: {data_source}")
                print(f"Original response: {response_str[:200]}...")
                print(f"Answer only: {answer_str[:200]}...")
                print(f"Has think: {think_data['has_think']}")
                print(f"Reward: {reward}")
                print("=" * 50)
                self.already_print_data_sources[data_source] = True
                self.num_examine -= 1
        
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra_info)}
        return reward_tensor