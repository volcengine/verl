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

import numpy as np
import torch

from recipe.fully_async_policy.utils import RolloutSample
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import postprocess_agent_loop_outputs
from verl.trainer.ppo.ray_trainer import compute_response_mask


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch: bool = False
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
        original_batch_dict = rs.original_batch_dict

        # 重建 DataProto
        original_batch_item = DataProto.from_single_dict(
            {
                **{k: v for k, v in original_batch_dict["batch"].items()},
                **{f"__{k}": v for k, v in original_batch_dict["non_tensor_batch"].items()},
            }
        )
        original_batch_item.meta_info.update(original_batch_dict["meta_info"])
        original_batch_list.append(original_batch_item)

    # 合并所有原始样本为一个批次
    if original_batch_list:
        original_batch = DataProto.from_items(original_batch_list)
    else:
        # 如果没有原始数据，创建空的 DataProto
        original_batch = DataProto.from_single_dict({})

    # 添加 UID
    uids = []
    for rs in rollout_samples:
        uids.append(f"uid_{rs.sample_id}")
    original_batch.non_tensor_batch["uid"] = np.array(uids, dtype=object)

    # 直接合并原始数据和生成结果，不需要 repeat
    # 因为队列中的每个 RolloutSample 都已经是独立的样本
    final_batch = original_batch.union(gen_batch_output)

    # 计算 response_mask（如果不存在）
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    # 简化的批次平衡逻辑（如果需要的话）
    if balance_batch and hasattr(config, "trainer") and getattr(config.trainer, "balance_batch", False):
        # 注意：这里简化了批次平衡逻辑，如果需要完整功能需要额外参数
        print("[BatchUtils] Batch balancing requested but simplified in static function")

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
