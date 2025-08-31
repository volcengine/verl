# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# Reference:
# - DAPO: An Open-Source LLM Reinforcement Learning System at Scale
#   Paper: https://arxiv.org/abs/2503.14476
# - This implementation references the ReTool implementation: recipe/retool/ in VERL codebase
import importlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from verl import DataProto


class DynamicFilter:
    """Unified class for handling dynamic filtering during training with state management."""

    def __init__(self, config):
        """Initialize the dynamic filter.

        Args:
            config: configuration from ray_trainer
        """
        # Configuration attributes
        self.metric = config.algorithm.filter_groups.metric
        self.filter_kwargs = config.algorithm.filter_groups.filter_kwargs
        self.custom_filter_func = None
        self.filter_function = config.algorithm.filter_groups.filter_function

        # State attributes
        self.num_gen_batches: int = 0
        self.num_prompt_in_batch: int = 0
        self.accumulated_batch: Optional[DataProto] = None
        self.reward_step: int = 0

        assert not config.reward_model.launch_reward_fn_async, (
            "Dynamic filter has not supported async reward function yet."
        )

        if self.filter_function:
            # Import custom filter function
            module_path, func_name = self.filter_function.rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.custom_filter_func = getattr(module, func_name)

    def clear(self) -> None:
        """Reset all state variables for the next training step."""
        if self.num_gen_batches > 0:
            print(f"Dynamic Filter: Used {self.num_gen_batches} generation batches to complete this step")

        self.num_gen_batches = 0
        self.num_prompt_in_batch = 0
        self.accumulated_batch = None
        self.reward_step = 0

    def increment_reward_step(self, global_step) -> bool:
        """Increment the reward step if it's less than the global step."""
        if self.reward_step < global_step:
            self.reward_step += 1
            return True
        return False

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

    def process_batch_with_filtering(
        self, batch: DataProto, config
    ) -> tuple[DataProto, bool]:
        """Process a batch with dynamic filtering and accumulation logic.

        Args:
            batch: The input batch to process
            config: configuration from ray_trainer

        Returns:
            tuple: (processed_batch, should_continue)
                - processed_batch: The batch ready for training (None if should continue)
                - should_continue: True if more batches are needed, False if ready for training
        """
        # Apply filtering to the batch - inline the filtering logic
        uids = batch.non_tensor_batch["uid"]
        token_level_scores = batch.batch["token_level_scores"]
        train_batch_size = config.data.train_batch_size
        max_num_gen_batches = config.algorithm.filter_groups.max_num_gen_batches
        rollout_n = config.actor_rollout_ref.rollout.n

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
        self.add_prompts(kept_prompts_this_batch)
        self.accumulate_batch(filtered_batch)

        # Check if we have enough prompts or reached max generation batches
        if (
            self.num_prompt_in_batch < train_batch_size
            and self.num_gen_batches < max_num_gen_batches
        ):
            return None, True  # Continue collecting more batches

        # If we reached max generation batches but still don't have enough prompts,
        # repeat batch content to fill the deficit
        if self.num_gen_batches >= max_num_gen_batches:
            if self.num_prompt_in_batch == 0:
                raise ValueError("No prompts collected in the generation batch,consider increasing max_num_gen_batches or rollout.n")
            prompt_deficit = train_batch_size - self.num_prompt_in_batch
            repeated_batch = self.accumulated_batch[: prompt_deficit * rollout_n]
            final_batch = DataProto.concat([self.accumulated_batch, repeated_batch])
        else:
            final_batch = self.accumulated_batch

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
