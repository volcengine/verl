# Copyright 2023-2024 SGLang Team
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
import importlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from verl import DataProto


@dataclass
class DynamicFilterState:
    """State tracking for dynamic filtering during batch processing."""

    num_gen_batches: int = 0
    num_prompt_in_batch: int = 0
    accumulated_batch: Optional[DataProto] = None

    def reset(self) -> None:
        """Reset all state variables for the next training step."""
        self.num_gen_batches = 0
        self.num_prompt_in_batch = 0
        self.accumulated_batch = None

    def increment_gen_batches(self) -> None:
        """Increment the generation batch counter."""
        self.num_gen_batches += 1

    def add_prompts(self, count: int) -> None:
        """Add to the prompt count."""
        self.num_prompt_in_batch += count

    def accumulate_batch(self, batch: DataProto) -> None:
        """Accumulate a batch, concatenating with existing if present."""
        self.accumulated_batch = (
            batch if self.accumulated_batch is None else DataProto.concat([self.accumulated_batch, batch])
        )


@dataclass
class DynamicFilterManager:
    """Manager class for handling dynamic filtering during training."""

    def __init__(self, filter_function: Optional[str] = None, metric: str = "seq_reward", **filter_kwargs):
        """Initialize the filter manager.

        Args:
            filter_function: Path to custom filter function (e.g., "my_module.my_filter_func").
                           If None, uses built-in mixed rewards filter.
            metric: Metric to use for filtering.
            **filter_kwargs: Additional arguments for the custom filter function.
        """
        self.metric = metric
        self.filter_kwargs = filter_kwargs
        self.custom_filter_func = None

        if filter_function:
            # Import custom filter function
            module_path, func_name = filter_function.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_filter_func = getattr(module, func_name)



    def process_batch_with_filtering(
        self,
        batch: DataProto,
        dynamic_filter_state: "DynamicFilterState",
        train_batch_size: int,
        max_num_gen_batches: int,
        rollout_n: int,
    ) -> tuple[DataProto, bool]:
        """Process a batch with dynamic filtering and accumulation logic.
        
        Args:
            batch: The input batch to process
            dynamic_filter_state: State object tracking filtering progress
            train_batch_size: Target number of prompts for training
            max_num_gen_batches: Maximum number of generation batches allowed
            rollout_n: Number of rollouts per prompt
            
        Returns:
            tuple: (processed_batch, should_continue)
                - processed_batch: The batch ready for training (None if should continue)
                - should_continue: True if more batches are needed, False if ready for training
        """
        # Apply filtering to the batch - inline the filtering logic
        uids = batch.non_tensor_batch["uid"]
        token_level_scores = batch.batch["token_level_scores"]

        # Handle metric calculation
        if self.metric == "seq_final_reward":
            raise ValueError("seq_final_reward is not supported for dynamic filter")
        elif self.metric == "seq_reward":
            # Calculate seq_reward if not already in batch
            if "seq_reward" not in batch.non_tensor_batch:
                batch.non_tensor_batch["seq_reward"] = token_level_scores.sum(dim=-1).numpy()

        # Group by prompt UID and collect metric values
        prompt_uid_to_metric_vals = defaultdict(list)
        for uid, metric_val in zip(uids, batch.non_tensor_batch[self.metric], strict=False):
            prompt_uid_to_metric_vals[uid].append(metric_val)

        # Apply filtering
        if not self.custom_filter_func:
            raise ValueError(
                "No filter function configured. Please specify filter_function in filter_groups config. "
                "For the original mixed rewards filter, use 'verl.utils.filtering.dynamic_filtering.keep_mixed_reward'"
            )

        kept_prompt_uids = []
        for prompt_uid, metric_vals in prompt_uid_to_metric_vals.items():
            should_keep = self.custom_filter_func(metric_vals, **self.filter_kwargs)
            if should_keep:
                kept_prompt_uids.append(prompt_uid)

        # Find indices of kept trajectories
        kept_traj_idxs = []
        for i, uid in enumerate(uids):
            if uid in kept_prompt_uids:
                kept_traj_idxs.append(i)

        kept_prompts_this_batch = len(kept_prompt_uids)
        
        # Filter the batch and update state
        filtered_batch = batch[kept_traj_idxs]
        dynamic_filter_state.add_prompts(kept_prompts_this_batch)
        dynamic_filter_state.accumulate_batch(filtered_batch)
        
        # Check if we have enough prompts or reached max generation batches
        if (
            dynamic_filter_state.num_prompt_in_batch < train_batch_size
            and dynamic_filter_state.num_gen_batches < max_num_gen_batches
        ):
            return None, True  # Continue collecting more batches
        
        # If we reached max generation batches but still don't have enough prompts,
        # repeat batch content to fill the deficit
        if dynamic_filter_state.num_gen_batches >= max_num_gen_batches:
            prompt_deficit = train_batch_size - dynamic_filter_state.num_prompt_in_batch
            repeated_batch = dynamic_filter_state.accumulated_batch[
                : prompt_deficit * rollout_n
            ]
            final_batch = DataProto.concat([dynamic_filter_state.accumulated_batch, repeated_batch])
        else:
            final_batch = dynamic_filter_state.accumulated_batch
        
        # Align the batch to the expected trajectory batch size
        traj_bsz = train_batch_size * rollout_n
        aligned_batch = final_batch[:traj_bsz]
        
        return aligned_batch, False  # Ready for training


def keep_mixed_reward(metric_vals: list[float | int], **kwargs) -> bool:
    """Original mixed rewards filter: keeps prompts with both positive and non-positive rewards.

    This is the original filtering logic that filters out prompts that are all positive
    OR all non-positive, keeping only prompts with mixed reward values.

    Args:
        metric_vals: List of metric values for samples from the same prompt
        **kwargs: Additional arguments (unused)

    Returns:
        bool: True if prompt should be kept (has mixed rewards), False otherwise

    Example configuration:
        algorithm:
          filter_groups:
            enable: true
            filter_function: "verl.utils.filtering.dynamic_filtering.keep_mixed_reward"
    """
    if len(metric_vals) <= 1:
        return True  # Always keep single samples

    metric_vals = np.array(metric_vals)
    all_positive = np.all(metric_vals > 0)
    all_non_positive = np.all(metric_vals <= 0)

    # Keep prompt only if it has both positive and non-positive values
    return not (all_positive or all_non_positive)
