# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import postprocess_agent_loop_outputs
from verl.trainer.ppo.ray_trainer import compute_response_mask


# Calculate the number of samples needed
def calculate_one_step_size(minimal_bsz, ppo_mini_batch_size):
    return minimal_bsz * ppo_mini_batch_size


@dataclass
class RolloutSample:
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # AgentLoopOutput from generation
    agent_loop_output: Any  # AgentLoopOutput

    # Metadata
    sample_id: str
    epoch: int
    rollout_n_index: int  # Index within the rollout.n repetitions (0, 1, ..., n-1)
    original_sample_index: int  # Index of the original sample before repetition

    # Processing metadata
    processing_time: float
    generation_timestamp: float
    param_version: int


def prepare_single_generation_data(batch_dict, global_steps) -> DataProto:
    """
    类似 ray_trainer._prepare_generate_batch 的逻辑，但针对单个样本
    分离出用于生成的数据和需要保留的原始数据

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    # 创建完整的 DataProto
    full_batch = DataProto.from_single_dict(batch_dict)

    # batch : TensorDict { input_ids, attention_mask, position_ids}
    # non_tensor_batch: raw_prompt_ids, raw_prompt,
    #                   multi_modal_data, tools_kwargs, interaction_kwargs, index, agent_name,
    #                   data_source, ability, reward_model
    # meta_info: {}

    # 定义需要传递给生成服务器的字段
    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

    full_batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
    )
    # 添加全局步数到生成数据
    full_batch.meta_info["global_steps"] = global_steps

    return full_batch


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    从 RolloutSample 对象中组装批次，类似 ray_trainer 的 _post_generate_batch 逻辑

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    # 直接处理 RolloutSample 对象
    processing_times = [rs.processing_time for rs in rollout_samples]

    # 第一步：从 AgentLoopOutput 创建生成结果的 DataProto
    agent_loop_outputs = [rs.agent_loop_output for rs in rollout_samples]
    gen_batch_output = postprocess_agent_loop_outputs(agent_loop_outputs, tokenizer, config)

    # 第二步：重建原始 batch 信息
    # 每个 RolloutSample 都是独立的，直接按顺序重建原始数据
    original_batch_list = []
    for rs in rollout_samples:
        item = rs.full_batch.to_items()[0]
        original_batch_list.append(item)

    # print("=" * 300)
    # print(original_batch_list)

    # 合并所有原始样本为一个批次
    if original_batch_list:
        original_batch = DataProto.from_items(original_batch_list)
    else:
        # 如果没有原始数据，创建空的 DataProto
        original_batch = DataProto.from_single_dict({})

    # print("=" * 300)
    # print(original_batch)

    # 添加 UID
    uids = []
    for rs in rollout_samples:
        uids.append(f"uid_{rs.sample_id}")
    original_batch.non_tensor_batch["uid"] = np.array(uids, dtype=object)

    # 直接合并原始数据和生成结果，不需要 repeat
    # 因为队列中的每个 RolloutSample 都已经是独立的样本
    if original_batch.batch is None:
        final_batch = gen_batch_output
        # 将 original_batch 的 non_tensor_batch 和 meta_info 合并到 final_batch
        for key, value in original_batch.non_tensor_batch.items():
            final_batch.non_tensor_batch[key] = value
        final_batch.meta_info.update(original_batch.meta_info)

    # 计算 response_mask（如果不存在）
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # 计算全局有效 token 数
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    # 收集统计信息和元数据（直接从 RolloutSample 中获取）
    param_versions = [rs.param_version for rs in rollout_samples]
    sample_timestamps = [rs.generation_timestamp for rs in rollout_samples]

    # 创建 meta_info
    final_batch.meta_info.update(
        {
            "rollout_param_versions": param_versions,
            "sample_timestamps": sample_timestamps,
            "avg_processing_time": np.mean(processing_times) if processing_times else 0,
            "max_processing_time": np.max(processing_times) if processing_times else 0,
            "param_version_diversity": len(set(param_versions)) if param_versions else 0,
            "avg_sample_age": np.mean([time.time() - ts for ts in sample_timestamps]) if sample_timestamps else 0,
            "assembly_time": time.time() - start_time,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch
