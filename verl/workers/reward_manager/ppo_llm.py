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

import torch
import re


from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from .ppo_rewards import compute_llm_scores


@register("ppo_llm")
class PPOLLMRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key=None,
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn

        # 1. 取出 extra_info 列表
        extra_info_list = data.non_tensor_batch.get("extra_info")

            
        
        # if "rm_scores" in data.batch.keys():
        #     # 有一个全0的reward
        #     print(f"[DEBUG] rm_scores exisits! Value of first item: {data.batch['rm_scores'][0]}", flush=True)
            
        #     raise RuntimeError("【调试】完蛋咯！")
            
        #     if return_dict:
        #         reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        #         reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
        #         return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        #     else:
        #         return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # 所有得分
        all_scores = [0.0] * len(data)
 
        # 记录规则匹配失败的数据，后续通过大模型判断正误
        llm_tasks = []
        
        # data_item: DataProtoItem
        for i, data_item in enumerate(data):
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            # 获取 ground truth
            # Non-Tensor Keys: ['extra_info', 'uid', '__num_turns__', 'raw_prompt']
            ground_truth = data_item.non_tensor_batch["extra_info"]["answer"]

            # 提取模型所给答案（如果提取不到怎么办?）
            resp_str = response_str.split('</think>')[-1].strip()
            match = re.search(r'\\boxed\{(.*?)\}', resp_str)
            if match:
                answer = match.group(1).strip()
            else:
                answer = ""

            # TODO: 计算score
            # 1. 尝试通过规则匹配
            r = 1.0 if answer == ground_truth else 0.0 
            
            if r == 1.0:
                all_scores[i] = r
            else:
                llm_tasks.append({
                    "index": i,
                    "question": prompt_str.split("<｜User｜>")[-1].split("<｜Assistant｜>")[0],
                    "stu_resp": resp_str,
                    "ground_truth": ground_truth
                })

        print(f"[DEBUG]: 需要大模型打分的数据总计 {len(llm_tasks)} 条")
        print(f"[DEBUG]: 总共 {len(data)} 条")
        print(f"第一条打分数据查看  {llm_tasks[0]} ")
        
        # 2. 对规则匹配失败的数据，请求大模型计算匹配reward
        scores = compute_llm_scores(llm_tasks, len(llm_tasks))

        # 加入大模型评估结果
        for score in scores:
            all_scores[score["index"]] = score["score"]
            
        for i, data_item in enumerate(data):   
            reward = all_scores[i]

            reward_tensor[i, valid_response_length - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor