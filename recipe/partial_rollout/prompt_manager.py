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
from collections import deque
from dataclasses import dataclass, is_dataclass
from typing import Any, Optional
import uuid
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

import asyncio
import numpy as np
import torch
import ray

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.ray_trainer import compute_response_mask

import logging
logger = logging.getLogger(__file__)
logger.setLevel("INFO")



@dataclass
class RolloutPrompt:
    """Enhanced rollout prompt (with n rollout samples) containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: DataProto

    # AgentLoopOutput from generation
    agent_loop_output_list: list[AgentLoopOutput]   # length: n

    # Metadata
    prompt_id: str
    epoch: int

    # Processing metadata
    processing_times: list[float]   # length: n
    tool_calls: list[float]         # length: n
    param_version: int
    param_version_start: list[int]  # length: n
    param_version_end: list[int]    # length: n
    rollout_status: dict[str, Any]
    original_batch: DataProto


def dict_of_list_to_list_of_dict(metrics: dict) -> list[dict]:
    """
    Convert:
        {k: [v1, v2, ...]}
    to:
        [{k: v1}, {k: v2}, ...]
    """
    if not metrics:
        return []

    keys = list(metrics.keys())
    length = len(next(iter(metrics.values())))

    for k, v in metrics.items():
        assert len(v) == length, f"Length mismatch for key '{k}'"

    return [
        {k: metrics[k][i] for k in keys}
        for i in range(length)
    ]


def assemble_batch_from_rollout_prompts(rollout_prompts: list[RolloutPrompt], current_param_version: int = None) -> DataProto:
    """
    Assemble gen_batch_output from RolloutPrompt objects
    Assembles batches from RolloutPrompt objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_prompts: List of RolloutPrompt objects
        current_param_version: Current parameter version
    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_prompts is empty
    """
    try:
        start_time = time.time()

        if not rollout_prompts:
            print("[Warning!!!] Empty rollout_prompts provided for batch assembly")
            return DataProto(batch=TensorDict({}, batch_size=(0,)), meta_info={})

        print(f"[BatchUtils] Assembling batch from {len(rollout_prompts)} RolloutPrompt objects")

        rollout_prompts_batch = []
        processing_times = []
        tool_calls = []
        rollout_status = rollout_prompts[0].rollout_status
        # Add a prefix to all rollout_status keys
        rollout_status = {f"partial_rollout/{key}": value for key, value in rollout_status.items()}

        for rp in rollout_prompts:
            rollout_prompts_batch.append(rp.full_batch)
            
        final_batch = DataProto.concat(rollout_prompts_batch)

        # Calculate response_mask (if not present)
        if "response_mask" not in final_batch.batch.keys():
            final_batch.batch["response_mask"] = compute_response_mask(final_batch)

        # Calculate the global valid token number
        if "attention_mask" in final_batch.batch:
            final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

        processing_times = final_batch.non_tensor_batch["processing_times"]
        tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
        # Collect statistics

        processing_time_stats = {
            "processing_time/avg": np.mean(processing_times),
            "processing_time/max": np.max(processing_times),
            "processing_time/min": np.min(processing_times),
            "processing_time/tp50": np.percentile(processing_times, 50),
            "processing_time/tp99": np.percentile(processing_times, 99),
            "processing_time/tp95": np.percentile(processing_times, 95),
        }
        tool_calls_stats = {}
        if len(tool_calls) > 0:
            tool_calls_stats = {
                # "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
                # "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
                # "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
            }
        processing_time_stats = {f"partial_rollout/{key}": value for key, value in processing_time_stats.items()}

        param_version_start = final_batch.non_tensor_batch["param_version_start"]
        param_version_end = final_batch.non_tensor_batch["param_version_end"]
        param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
        num_diff0 = param_version_diff.count(0)
        partial_stats = {
            "partial_rollout/partial/total_partial_num": len(param_version_diff) - num_diff0,
            "partial_rollout/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
            "partial_rollout/partial/max_partial_span": max(param_version_diff),
        }
        staleness_stats = {}
        if current_param_version is not None:
            staleness = [current_param_version - version_start for version_start in param_version_start]
            staleness_stats.update({
                "partial_rollout/partial/staleness_max": np.max(staleness),
                "partial_rollout/partial/staleness_min": np.min(staleness),
                "partial_rollout/partial/staleness_avg": np.mean(staleness),
                "partial_rollout/partial/staleness_tp50": np.percentile(staleness, 50),
                "partial_rollout/partial/staleness_tp99": np.percentile(staleness, 99),
                "partial_rollout/partial/staleness_tp95": np.percentile(staleness, 95),
            })
        # add meta_info
        param_versions = [rp.param_version for rp in rollout_prompts]
        trajectorys_param_versions = final_batch.non_tensor_batch["param_version_end"]

        final_batch.meta_info.update(
            {
                "rollout_param_versions": param_versions,
                "param_version_diversity": len(set(param_versions)) if param_versions else 0,
                "trajectory_param_versions": trajectorys_param_versions,
                **processing_time_stats,
                **rollout_status,
                **partial_stats,
                **staleness_stats,
                **tool_calls_stats,
            }
        )

        final_batch.meta_info["metrics"] = dict_of_list_to_list_of_dict(final_batch.meta_info["metrics"])

        print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    except Exception as e:
        logger.error(f"[BatchUtils] Batch assembly failed: {e}")
        breakpoint()
        raise e

    return final_batch



@ray.remote
class RolloutPromptManager:
    """
    Ray-based asynchronous rollout prompt manager for communication between AgentLoop and Trainer
    """

    def __init__(self, config: DictConfig, tokenizer, processor, dataloader: DataLoader):
        self.config = config
        self.cancellation_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self.epoch = 0
        self.current_param_version = 0
        self.ongoing_set = set()
        self.pending_queue = deque()
        self.done_queue = deque()

        from recipe.partial_rollout.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        self.dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        self.sampler = create_rl_sampler(config.data, self.dataset)
        self.dataloader = StatefulDataLoader(
            dataset=self.dataset,
            batch_size=config.data.get("gen_batch_size", config.data.train_batch_size),
            num_workers=config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=self.sampler,
        )
        
        self.dataiter = iter(self.dataloader)
        self.is_dataiter_exhausted = False
    
    def on_epoch_start(self, epoch: int):
        """On epoch start for the rollout prompt manager."""
        self.epoch = epoch
        if self.is_dataiter_exhausted:
            self.dataiter = iter(self.dataloader)
            self.is_dataiter_exhausted = False

    def prepare_generation(self, param_version: int):
        """Prepare generation for the rollout prompt manager."""
        self.cancellation_event.clear()
        self.ongoing_set.clear()
        self.current_param_version = param_version

    def check_generation_once(self, num_rollout_prompts: int) -> bool:
        """Check generation for the rollout prompt manager."""
        done = len(self.done_queue) >= num_rollout_prompts or \
            (len(self.ongoing_set) == 0 and len(self.pending_queue) == 0 and self.is_dataiter_exhausted)

        if done:
            logger.info(
                f"[RolloutPromptManager] check_generation_once:\n"
                f"  - num_rollout_prompts: {num_rollout_prompts}\n"
                f"  - num_done_queue: {len(self.done_queue)}\n"
                f"  - num_pending_queue: {len(self.pending_queue)}\n"
                f"  - num_ongoing_set: {len(self.ongoing_set)}\n"
            )
            self.cancellation_event.set()
        return done
    
    def check_generation_post_state(self, num_rollout_prompts: int) -> bool:
        """Check generation post state for the rollout prompt manager."""
        logger.info(
            "==========================================================\n"
            f"[RolloutPromptManager] check_generation_post_state:\n"
            f"  - num_rollout_prompts: {num_rollout_prompts}\n"
            f"  - num_done_queue: {len(self.done_queue)}\n"
            f"  - num_pending_queue: {len(self.pending_queue)}\n"
            f"  - num_ongoing_set: {len(self.ongoing_set)}\n"
            "==========================================================\n"
        )
        return (
            len(self.ongoing_set) == 0 and
            len(self.done_queue) >= num_rollout_prompts
        )

    def pull_done_prompts(self, num_rollout_prompts: int) -> list[DataProto]:
        """Pull done prompts from the rollout prompt manager."""
        n = min(num_rollout_prompts, len(self.done_queue))
        return [
            assemble_batch_from_rollout_prompts(
                [self.done_queue.popleft() for _ in range(n)],
                self.current_param_version,
            )
        ]

    def push_done_prompt(self, rollout_prompt: RolloutPrompt, is_cancel: bool = False):
        """Push done prompts to the rollout prompt manager."""
        try:
            if is_cancel:
                self.pending_queue.appendleft(rollout_prompt)
            else:
                rollout_prompt.full_batch.non_tensor_batch["uid"] = np.array(
                    [f"uid_{rollout_prompt.prompt_id}"] * len(rollout_prompt.full_batch), dtype=object
                )
                rollout_prompt.full_batch.union(rollout_prompt.original_batch)
                rollout_prompt.param_version = self.current_param_version
                param_version_start = rollout_prompt.full_batch.non_tensor_batch["param_version_start"]
                param_version_end = rollout_prompt.full_batch.non_tensor_batch["param_version_end"]
                param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
                if max(param_version_diff) < 10:
                    self.done_queue.append(rollout_prompt)

                assert rollout_prompt.prompt_id in self.ongoing_set, f"prompt {rollout_prompt.prompt_id} not in ongoing_set"
                
            self.ongoing_set.remove(rollout_prompt.prompt_id)
        except Exception as e:
            logger.error(f"[RolloutPromptManager] push_done_prompt: {e}")
            breakpoint()

    def pull_pending_prompts(self, num_rollout_prompts: int) -> list[RolloutPrompt]:
        """Pull pending prompts from the rollout prompt manager."""
        try:
            pending_prompts = []

            while len(pending_prompts) < num_rollout_prompts:
                if self.is_dataiter_exhausted or self.cancellation_event.is_set():
                    break

                n = min(num_rollout_prompts - len(pending_prompts), len(self.pending_queue))
                if len(self.pending_queue) > 0:
                    pending_prompts.extend([self.pending_queue.popleft() for _ in range(n)])
                else:
                    try:
                        batch_dict = next(self.dataiter)
                        batch = DataProto.from_single_dict(batch_dict)
                        batch: list[DataProto] = batch.chunk(batch.batch.size(0))
                        self.pending_queue.extend(self._prepare_single_rollout_prompt(data) for data in batch)
                    except StopIteration:
                        self.is_dataiter_exhausted = True
            
            for prompt in pending_prompts:
                assert prompt.prompt_id not in self.ongoing_set, f"prompt {prompt.prompt_id} already in ongoing_set"
                self.ongoing_set.add(prompt.prompt_id)
        
        except Exception as e:
            logger.error(f"[RolloutPromptManager] pull_pending_prompts: {e}")
            breakpoint()
            
        return pending_prompts


    def _prepare_single_rollout_prompt(self, data: DataProto) -> RolloutPrompt:
        """Prepare a single rollout prompt."""
        import copy
        
        config = self.config
        original_batch = copy.deepcopy(data)

        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & original_batch.non_tensor_batch.keys()

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(original_batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = original_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )
        
        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(original_batch.non_tensor_batch)

        # Setting selected agent, that supports partial
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            gen_batch.non_tensor_batch["agent_name"] = np.array(
                ["partial_tool_agent"] * len(gen_batch), dtype=object
            )
        else:
            gen_batch.non_tensor_batch["agent_name"] = np.array(
                ["partial_single_turn_agent"] * len(gen_batch), dtype=object
            )

        original_batch = original_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
        gen_batch = gen_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
        gen_batch.non_tensor_batch["param_version"] = [self.current_param_version] * len(gen_batch)

        return RolloutPrompt(
            full_batch=gen_batch,
            agent_loop_output_list=[None] * self.config.actor_rollout_ref.rollout.n,
            prompt_id=f"prompt_{uuid.uuid4()}",
            epoch=self.epoch,
            param_version=self.current_param_version,   # finish param version
            param_version_start=[],                     # len()=n, start param version
            param_version_end=[],                       # len()=n, end param version
            processing_times=[],
            tool_calls=[],
            rollout_status={},
            original_batch=original_batch,
        )


