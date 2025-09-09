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
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

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
    agent_loop_output_list: list[Any]  # AgentLoopOutput

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    processing_times: list[float]
    param_version: int
    rollout_status: dict[str, Any]


@dataclass
class ValidateMetrics:
    timing_raw: dict[str, Any]
    metrics: dict[str, Any]
    global_steps: Optional[int] = None
    param_version: Optional[int] = None


def prepare_single_generation_data(batch_dict, global_steps, rollout_n) -> DataProto:
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

    # 设置使用支持partial的agent
    full_batch.non_tensor_batch["agent_name"] = np.array(["partial_single_turn_agent"] * len(full_batch), dtype=object)

    # 添加全局步数到生成数据
    full_batch.meta_info["global_steps"] = global_steps
    full_batch = full_batch.repeat(repeat_times=rollout_n, interleave=True)
    return full_batch


def process_rollout_log_probs(data_proto: DataProto, rollout_log_probs: list[list[float]]) -> torch.Tensor:
    """
    根据 DataProto 中的 mask 逻辑处理 rollout_log_probs
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]

    Args:
        data_proto: 包含 batch 信息的 DataProto 对象
        rollout_log_probs: 二维列表，每个子列表包含一个样本的 log_probs

    Returns:
        torch.Tensor: 处理后的 log_probs tensor，形状为 [bsz, response_length]
    """

    batch = data_proto.batch
    response_mask = batch["response_mask"]
    bsz, response_length = response_mask.shape

    # 初始化结果 tensor
    rollout_log_probs_tensor = torch.zeros((bsz, response_length), dtype=torch.float32) - 1

    for i, log_probs_seq in enumerate(rollout_log_probs):
        # 获取当前样本的有效长度（mask 中为 1 的位置数量）
        valid_length = response_mask[i].sum().item()

        # 确保 log_probs_seq 的长度不超过有效长度
        actual_length = min(len(log_probs_seq), valid_length)

        # 将 log_probs 填入对应位置
        if actual_length > 0:
            rollout_log_probs_tensor[i, :actual_length] = torch.tensor(log_probs_seq[:actual_length])

    rollout_log_probs_tensor = rollout_log_probs_tensor.to(torch.float32)
    return rollout_log_probs_tensor


def merge_rollout_sample(config, tokenizer, rs: RolloutSample):
    # 第一步：从 AgentLoopOutput 创建生成结果的 DataProto
    gen_batch_output = postprocess_agent_loop_outputs(rs.agent_loop_output_list, tokenizer, config)
    rollout_log_probs = [x.log_probs for x in rs.agent_loop_output_list]
    rollout_log_probs = process_rollout_log_probs(gen_batch_output, rollout_log_probs)
    gen_batch_output.batch["rollout_log_probs"] = rollout_log_probs.to(torch.float32)

    # 第二步：添加 uid
    rs.full_batch.non_tensor_batch["uid"] = np.array([f"uid_{rs.sample_id}"] * len(rs.full_batch), dtype=object)

    # 第二步：合并batch
    # 将 original_batch 的 non_tensor_batch 和 meta_info 合并到 final_batch
    for key, value in rs.full_batch.non_tensor_batch.items():
        gen_batch_output.non_tensor_batch[key] = value
    gen_batch_output.meta_info.update(rs.full_batch.meta_info)

    # 第三步，设置 full_batch
    rs.full_batch = gen_batch_output
    rs.processing_times = []
    for agent_loop in rs.agent_loop_output_list:
        rs.processing_times.append(agent_loop.metrics.generate_sequences)

    # 第四步，清空 agent_loop_output_list
    rs.agent_loop_output_list = []

    return rs


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

    rollout_samples_batch = []
    processing_times = []
    rollout_status = rollout_samples[0].rollout_status
    # 为 rollout_status 的所有 key 添加前缀
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        rollout_samples_batch.append(rs.full_batch)
        processing_times.extend(rs.processing_times)
    final_batch = DataProto.concat(rollout_samples_batch)

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

    processing_time_stats = {
        "avg_processing_time": np.mean(processing_times),
        "max_processing_time": np.max(processing_times),
        "min_processing_time": np.min(processing_times),
        "tp50_processing_time": np.percentile(processing_times, 50),  # 中位数
        "tp99_processing_time": np.percentile(processing_times, 99),  # 99百分位
        "tp95_processing_time": np.percentile(processing_times, 95),  # 95百分位也很有用
    }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    # 创建 meta_info
    final_batch.meta_info.update(
        {
            "rollout_param_versions": param_versions,
            "param_version_diversity": len(set(param_versions)) if param_versions else 0,
            **processing_time_stats,
            **rollout_status,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch


class MetricsAggregator:
    """Metrics aggregator, used to combine metrics from multiple training steps"""

    def __init__(self, total_gpus: int):
        # Store all values ​​for each metric
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        # Store the number of samples at each step for weighted averaging
        self.sample_counts: list[int] = []
        # Store the timestamp of each step for time-related calculations
        self.timestamps: list[float] = []
        # Step Count
        self.step_count = 0
        # total num gpus used
        self.total_gpus = total_gpus

        # Metric aggregation rule configuration
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, dict[str, list[str]]]:
        """Initialize metrics aggregation rules"""
        return {
            # Time-Based metrics, can add metrics here
            "time_sum": ["perf/time_per_step"],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int, timestamp: float = None):
        """Adding a single-step metrics"""
        if timestamp is None:
            timestamp = time.time()

        self.sample_counts.append(sample_count)
        self.timestamps.append(timestamp)
        self.step_count += 1

        # Store all metrics values
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.metric_values[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.metric_values[key].append(float(value.item()))

    def _get_aggregation_type(self, metric_name: str) -> str:
        """Determine the aggregation type based on the metric name"""
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if any(keyword in metric_lower for keyword in ["timing_s/"]):
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["mean", "avg", "average"]):
            return "avg"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg"]):
            return "weighted_avg"

        import warnings

        warnings.warn(
            f"No aggregation rule is matched in init_aggregation_rules. \
                      For metric {metric_name}, the 'avg' method is used"
        )
        return "avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        """Aggregating a single metric"""
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)

        if agg_type == "last":
            return values[-1]

        elif agg_type == "weighted_avg":
            # Weighted average
            if len(values) != len(self.sample_counts):
                # If the lengths do not match, use a simple average
                return sum(values) / len(values)

            total_samples = sum(self.sample_counts)
            if total_samples == 0:
                return sum(values) / len(values)

            weighted_sum = sum(v * c for v, c in zip(values, self.sample_counts, strict=False))
            return weighted_sum / total_samples

        elif agg_type == "sum" or agg_type == "time_sum":
            return sum(values)

        elif agg_type == "avg":
            return sum(values) / len(values)

        elif agg_type == "max":
            return max(values)

        elif agg_type == "min":
            return min(values)

        else:
            # Default average
            return sum(values) / len(values)

    def get_aggregated_metrics(self) -> dict[str, Any]:
        """aggregated metrics"""
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(metric_name, values)

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated)

        print(f"aggregated metrics done. cost {time.time() - t}")

        return aggregated

    def _special_metrics_aggergate(self, aggregated: dict[str, Any]) -> dict[str, Any]:
        """calculate special metrics"""

        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        REQUIRED_PERF_KEYS = {"perf/throughput", "perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        return aggregated

    def reset(self):
        """Reset Aggregator"""
        self.metric_values.clear()
        self.sample_counts.clear()
        self.timestamps.clear()
        self.step_count = 0

    def get_current_stats(self) -> dict[str, Any]:
        """Get statistics about the current aggregation state (for debugging)"""
        return {
            "step_count": self.step_count,
            "metric_count": len(self.metric_values),
            "total_samples": sum(self.sample_counts),
            "metric_names": list(self.metric_values.keys()),
        }
