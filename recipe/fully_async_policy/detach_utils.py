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
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.ray_trainer import compute_response_mask


def postprocess_agent_loop_outputs(inputs: list[AgentLoopOutput], tokenizer, config) -> DataProto:
    """Static method to postprocess a list of AgentLoopOutput into DataProto

    Args:
        inputs: List of AgentLoopOutput
        tokenizer: Tokenizer instance
        config: Configuration object

    Returns:
        DataProto: Processed batch data
    """
    # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
    # prompts: left pad
    # responses: right pad
    # input_ids: prompt + response
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

    # prompts
    tokenizer.padding_side = "left"
    outputs = tokenizer.pad(
        [{"input_ids": input.prompt_ids} for input in inputs],
        padding="max_length",
        max_length=config.actor_rollout_ref.rollout.prompt_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

    # responses
    tokenizer.padding_side = "right"
    outputs = tokenizer.pad(
        [{"input_ids": input.response_ids} for input in inputs],
        padding="max_length",
        max_length=config.actor_rollout_ref.rollout.response_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

    # response_mask
    outputs = tokenizer.pad(
        [{"input_ids": input.response_mask} for input in inputs],
        padding="max_length",
        max_length=config.actor_rollout_ref.rollout.response_length,
        return_tensors="pt",
        return_attention_mask=False,
    )
    response_mask = outputs["input_ids"]
    assert response_ids.shape == response_mask.shape, (
        f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
    )
    response_mask = response_mask * response_attention_mask

    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

    batch = TensorDict(
        {
            "prompts": prompt_ids,  # [bsz, prompt_length]
            "responses": response_ids,  # [bsz, response_length]
            "response_mask": response_mask,  # [bsz, response_length]
            "input_ids": input_ids,  # [bsz, prompt_length + response_length]
            "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
            "position_ids": position_ids,  # [bsz, prompt_length + response_length]
        },
        batch_size=len(input_ids),
    )

    num_turns = np.array([input.num_turns for input in inputs], dtype=np.int32)
    metrics = [input.metrics.model_dump() for input in inputs]
    return DataProto(batch=batch, non_tensor_batch={"__num_turns__": num_turns}, meta_info={"metrics": metrics})


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
    param_version_start: list[int]
    param_version_end: list[int]
    rollout_status: dict[str, Any]


@dataclass
class ValidateMetrics:
    """Metrics for validation"""

    timing_raw: dict[str, Any]
    metrics: Optional[dict[str, Any]] = None
    global_steps: Optional[int] = None
    param_version: Optional[int] = None


def prepare_single_generation_data(batch_dict, global_steps, rollout_n) -> DataProto:
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

    full_batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
    )

    # Setting agent - partial_single_turn_agent, that supports partial
    full_batch.non_tensor_batch["agent_name"] = np.array(["partial_single_turn_agent"] * len(full_batch), dtype=object)

    # Add global step count to generated data
    full_batch = full_batch.repeat(repeat_times=rollout_n, interleave=True)
    return full_batch


def process_rollout_log_probs(data_proto: DataProto, rollout_log_probs: list[list[float]]) -> torch.Tensor:
    """
    Process rollout_log_probs according to the mask in DataProto
    mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]

    Args:
        data_proto: A DataProto object containing batch information
        rollout_log_probs: A two-dimensional list, each sublist containing the log_probs of a sample

    Returns:
        torch.Tensor: The processed log_probs tensor, with shape: [bsz, response_length]
    """

    batch = data_proto.batch
    response_mask = batch["response_mask"]
    rollout_log_probs_tensor = torch.zeros(response_mask.shape, dtype=torch.float32) - 1

    for i, log_probs_seq in enumerate(rollout_log_probs):
        # Get the effective length of the current sample (the number of positions with 1 in the mask)
        valid_length = response_mask[i].sum().item()

        # Ensure that the length of log_probs_seq does not exceed the valid length
        actual_length = min(len(log_probs_seq), valid_length)

        # Fill log_probs into the corresponding position
        if actual_length > 0:
            rollout_log_probs_tensor[i, :actual_length] = torch.tensor(log_probs_seq[:actual_length])

    rollout_log_probs_tensor = rollout_log_probs_tensor.to(torch.float32)
    return rollout_log_probs_tensor


def merge_rollout_sample(config, tokenizer, rs: RolloutSample):
    """
    Supplement and refine the RolloutSample object,
    """
    # Step 1: Create a DataProto from the AgentLoopOutput to generate the result
    gen_batch_output = postprocess_agent_loop_outputs(rs.agent_loop_output_list, tokenizer, config)
    rollout_log_probs = [x.log_probs for x in rs.agent_loop_output_list]
    rollout_log_probs = process_rollout_log_probs(gen_batch_output, rollout_log_probs)
    gen_batch_output.batch["rollout_log_probs"] = rollout_log_probs.to(torch.float32)

    # Step 2: Add uid
    rs.full_batch.non_tensor_batch["uid"] = np.array([f"uid_{rs.sample_id}"] * len(rs.full_batch), dtype=object)

    # Step 2: Merge batches
    # Merge the non_tensor_batch and meta_info of original_batch into final_batch
    for key, value in rs.full_batch.non_tensor_batch.items():
        gen_batch_output.non_tensor_batch[key] = value
    gen_batch_output.meta_info.update(rs.full_batch.meta_info)

    # Step 3, set full_batch
    rs.full_batch = gen_batch_output
    rs.processing_times = []
    for agent_loop in rs.agent_loop_output_list:
        rs.processing_times.append(agent_loop.metrics.generate_sequences)
    rs.param_version_start = [agent_loop.param_version_start for agent_loop in rs.agent_loop_output_list]
    rs.param_version_end = [agent_loop.param_version_end for agent_loop in rs.agent_loop_output_list]
    # Step 4, clear agent_loop_output_list
    rs.agent_loop_output_list = []
    return rs


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

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
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        rollout_samples_batch.append(rs.full_batch)
        processing_times.extend(rs.processing_times)
    final_batch = DataProto.concat(rollout_samples_batch)

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    # Collect statistics
    param_versions = [rs.param_version for rs in rollout_samples]
    trajectorys_param_versions = [version for rs in rollout_samples for version in rs.param_version_end]

    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    param_version_diff = [abs(a - b) for a, b in zip(rs.param_version_end, rs.param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    final_batch.meta_info.update(
        {
            "rollout_param_versions": param_versions,
            "param_version_diversity": len(set(param_versions)) if param_versions else 0,
            "trajectory_param_versions": trajectorys_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
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
            "last": [
                "fully_async/count/total_generated_samples",
                "fully_async/count/stale_samples_processed",
                "fully_async/count/stale_trajectory_processed",
                "fully_async/count/current_param_version",
                "fully_async/count/dropped_stale_samples",
                "training/global_step",  # TODO change name to: total_step
            ],
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
            if isinstance(value, int | float | np.number):
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

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        # perf/throughput
        REQUIRED_PERF_KEYS = {"perf/throughput", "perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["trainer/idle_ratio"] = aggregated["timing_s/gen"] / aggregated["timing_s/step"]

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
