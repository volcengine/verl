# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""
EnvLoop for Isaac Server Mode with Batched Step and Pipeline Overlap

Key Features:
1. Batched Step: All trajectories in the same pipeline stage are batched together
   - Stage 0: [traj_0, traj_2, ...] → single batched step
   - Stage 1: [traj_1, traj_3, ...] → single batched step

2. Pipeline Overlap (Phase 2): Stage 0 sim overlaps with Stage 1 gen
   - While Isaac simulates Stage 0, rollout workers generate Stage 1 actions
   - Maximizes GPU utilization

Architecture:
    EnvLoopServer
        │
        ├── rollout_wg (multiple rollout workers for model inference)
        │
        └── env_wg (single EnvWorkerServer as coordinator)
                │
                └── IsaacMultiTaskServer (remote, manages all envs)

Data Flow:
    Reset:
        prompts (state_ids, task_ids)
            → EnvWorkerServer.reset_envs_to_state_ids()
            → Returns: obs + trajectory_keys (hash-based)
            → Store trajectory_keys for step calls

    Step (Batched):
        Pipeline Stage 0:
            - Collect all traj actions: [traj_0, traj_2, ...]
            - Batch: [N_envs × 8 chunks]
            - Single isaac.step() call
        Pipeline Stage 1:
            - Collect all traj actions: [traj_1, traj_3, ...]
            - Batch: [N_envs × 8 chunks]
            - Single isaac.step() call
"""

import asyncio
import logging
import os

import numpy as np
import ray
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnvLoopServer:
    """
    EnvLoop for Isaac Server Mode with Batched Step.

    Manages interactions between rollout workers (model inference) and
    a single EnvWorkerServer (coordinator) that connects to Isaac Multi-Task Server.

    Batching Strategy:
        - Trajectories are grouped by pipeline stage
        - All trajectories in the same stage are batched together
        - Single isaac.step() call per stage (instead of per-trajectory)
    """

    def __init__(self, env_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, config: DictConfig):
        """
        Initialize the EnvLoopServer.

        Args:
            env_wg (RayWorkerGroup): Environment worker group (single EnvWorkerServer).
            rollout_wg (RayWorkerGroup): Rollout worker group for model inference.
            config (DictConfig): YAML config.
        """
        self.env_wg = env_wg
        self.rollout_wg = rollout_wg
        self.config = config

        # Pipeline configuration
        self.stage_num = config.env.rollout.pipeline_stage_num  # Number of pipeline stages (e.g., 2)
        self.num_envs_per_traj = config.env.train.num_envs  # envs per trajectory (e.g., 8)
        self.action_dim = config.env.actor.model.action_dim
        self.num_action_chunks = config.env.actor.model.num_action_chunks
        self.max_interactions = config.env.train.max_episode_steps // self.num_action_chunks

        # Server mode: total_envs from config
        self.total_envs = config.env.train.get("total_envs", 128)

        # Calculate trajectory distribution
        # total_trajectories = total_envs / num_envs_per_traj
        self.total_trajectories = self.total_envs // self.num_envs_per_traj

        # Trajectories per pipeline stage
        # e.g., 4 trajectories, 2 stages → 2 trajectories per stage
        self.trajectories_per_stage = self.total_trajectories // self.stage_num

        # Envs per pipeline stage (for batched step)
        self.envs_per_stage = self.trajectories_per_stage * self.num_envs_per_traj

        # Trajectory keys (set during reset, used during step)
        self._trajectory_keys: list[str] = []

        # Pipeline stage to trajectory mapping
        # stage_0: [traj_0, traj_2, ...], stage_1: [traj_1, traj_3, ...]
        self._stage_to_traj_indices: dict[int, list[int]] = {}
        self._setup_stage_mapping()

        logger.info("EnvLoopServer initialized (Batched Mode):")
        logger.info(f"  Total envs: {self.total_envs}")
        logger.info(f"  Total trajectories: {self.total_trajectories}")
        logger.info(f"  Pipeline stages: {self.stage_num}")
        logger.info(f"  Trajectories per stage: {self.trajectories_per_stage}")
        logger.info(f"  Envs per stage (batch size): {self.envs_per_stage}")
        logger.info(f"  Max interactions: {self.max_interactions}")

        # Initialize workers
        self.env_wg.init_worker()
        self.env_wg.init_simulator()

    def _setup_stage_mapping(self):
        """
        Setup mapping from pipeline stage to trajectory indices.

        Distribution: round-robin assignment
        - Stage 0: traj_0, traj_2, traj_4, ...
        - Stage 1: traj_1, traj_3, traj_5, ...
        """
        for stage_id in range(self.stage_num):
            self._stage_to_traj_indices[stage_id] = []

        for traj_idx in range(self.total_trajectories):
            stage_id = traj_idx % self.stage_num
            self._stage_to_traj_indices[stage_id].append(traj_idx)

        logger.info(f"Stage to trajectory mapping: {self._stage_to_traj_indices}")

    def _get_traj_keys_for_stage(self, stage_id: int) -> list[str]:
        """Get all trajectory keys for a pipeline stage."""
        traj_indices = self._stage_to_traj_indices[stage_id]
        keys = []
        for traj_idx in traj_indices:
            start = traj_idx * self.num_envs_per_traj
            end = (traj_idx + 1) * self.num_envs_per_traj
            keys.extend(self._trajectory_keys[start:end])
        return keys

    def generate_sequences(self, prompts: DataProto, reset_future) -> DataProto:
        """
        Generate sequences through environment interaction.

        Args:
            prompts (DataProto): Input batch containing state_ids, task_ids, etc.
            reset_future: Future or list from env reset (depends on dispatch mode).

        Returns:
            DataProto: Output batch with collected trajectories.
        """
        # ONE_TO_ALL dispatch returns list of ObjectRefs, use ray.get()
        reset_results = ray.get(reset_future[0])

        # Debug: print reset_results structure
        print(f"[DEBUG reset_results] type: {type(reset_results)}", flush=True)
        try:
            print(f"[DEBUG reset_results] batch keys: {list(reset_results.batch.keys())}", flush=True)
        except (TypeError, AttributeError):
            print("[DEBUG reset_results] batch: None or empty", flush=True)
        if reset_results.non_tensor_batch:
            print(
                f"[DEBUG reset_results] non_tensor_batch keys: {list(reset_results.non_tensor_batch.keys())}",
                flush=True,
            )

        # Extract and store trajectory_keys from reset results (one key per env)
        self._trajectory_keys = reset_results.non_tensor_batch.get("trajectory_keys", [])
        print(f"[DEBUG] Reset returned {len(self._trajectory_keys)} trajectory_keys", flush=True)
        print(f"[DEBUG] Stage mapping: {self._stage_to_traj_indices}", flush=True)

        loop = asyncio.get_event_loop()
        self.rollout_wg.switch_to_rollout()
        output = loop.run_until_complete(self.run(prompts, reset_results))
        self.rollout_wg.switch_to_train()

        return output

    async def run(self, prompts: DataProto, reset_results: DataProto) -> DataProto:
        """
        Run the environment interaction loop with PIPELINE OVERLAP.

        Pipeline Strategy:
        - Stage 0 sim runs concurrently with Stage 1 generation
        - Stage 1 sim runs concurrently with Stage 0 generation (next step)
        - All stages run their loops in parallel via asyncio.gather

        Flow per stage (runs in parallel with other stages):
        1. For each step:
           a. Wait for generation result
           b. Send batched step to Isaac (async)
           c. Wait for sim result
           d. Start next generation (async, overlaps with other stage's sim)

        Args:
            prompts (DataProto): Contains initial state IDs and other settings.
            reset_results (DataProto): Observations from reset, including trajectory_keys.

        Returns:
            DataProto: A batch containing the complete trajectories.
        """
        initial_state_ids = prompts.non_tensor_batch["state_ids"]

        # Restructure reset observations by pipeline stage
        staged_obs = self._restructure_obs_by_stage(reset_results)

        print(
            f"[DEBUG run] stage_num: {self.stage_num}, trajectories_per_stage: {self.trajectories_per_stage}",
            flush=True,
        )

        # --- Trajectory storage (shared across stages) ---
        trajectories = {stage_id: [] for stage_id in range(self.stage_num)}

        # --- Start first generation for each stage (async, overlapped) ---
        rollout_futures = {}
        for stage_id in range(self.stage_num):
            trajectories[stage_id].append({})
            vla_input = staged_obs[stage_id]
            vla_input.meta_info = prompts.meta_info
            rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

        async def _stage_loop(stage_id: int):
            """
            Per-stage loop that runs in parallel with other stages.
            Overlap: While this stage does sim, other stages can do generation.
            """
            for step_idx in range(self.max_interactions):
                # === Wait for generation result ===
                action_result: DataProto = await asyncio.to_thread(rollout_futures[stage_id].get)
                trajectories[stage_id][-1]["action"] = action_result

                # === Batched environment step ===
                stage_keys = self._get_traj_keys_for_stage(stage_id)
                action_data = DataProto.from_dict(
                    non_tensors={
                        "actions": action_result.batch["action"].cpu().numpy(),
                        "trajectory_keys": stage_keys,
                    },
                    meta_info={
                        "stage_id": stage_id,
                        "step_idx": step_idx,
                        "max_steps": self.max_interactions,
                    },
                )

                # Send step request (will be batched with other stages at server)
                env_refs = self.env_wg.env_interact_step(action_data)
                env_result: DataProto = await asyncio.to_thread(ray.get, env_refs[0])

                # Store results
                trajectories[stage_id][-1]["rew"] = env_result.batch["rews"]
                trajectories[stage_id][-1]["done"] = env_result.batch["terminations"]

                # Prepare next observation
                next_obs = DataProto(
                    batch=env_result.batch.select("full_image", "state"),
                    non_tensor_batch={"task_descriptions": env_result.non_tensor_batch["task_descriptions"]},
                )

                # === Start next generation (async, overlaps with other stage's sim) ===
                if step_idx < self.max_interactions - 1:
                    trajectories[stage_id].append({})
                    vla_input = next_obs
                    vla_input.meta_info = prompts.meta_info
                    rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

        # === Run all stage loops in parallel ===
        await asyncio.gather(*[asyncio.create_task(_stage_loop(sid)) for sid in range(self.stage_num)])

        self.env_wg.finish_rollout()

        return self._collate_trajectories(trajectories, initial_state_ids, meta_info=prompts.meta_info)

    def _restructure_obs_by_stage(self, data_proto: DataProto) -> list[DataProto]:
        """
        Restructure flat observation data into per-pipeline-stage DataProto.

        Input: flat data ordered by trajectory [traj_0, traj_1, traj_2, traj_3, ...]
        Output: list of DataProto, one per pipeline stage
            - staged_obs[0] = [traj_0, traj_2, ...] (stage 0)
            - staged_obs[1] = [traj_1, traj_3, ...] (stage 1)
        """
        # Split into per-trajectory chunks
        traj_chunks = data_proto.chunk(self.total_trajectories)

        staged_data = []
        for stage_id in range(self.stage_num):
            traj_indices = self._stage_to_traj_indices[stage_id]
            stage_chunks = [traj_chunks[idx] for idx in traj_indices]
            staged_data.append(DataProto.concat(stage_chunks))

        # Debug
        try:
            input_size = data_proto.batch["full_image"].shape[0]
            print(f"[DEBUG _restructure] Input size: {input_size}", flush=True)
        except (KeyError, TypeError, AttributeError):
            print("[DEBUG _restructure] Input size: None", flush=True)

        for stage_id, stage_data in enumerate(staged_data):
            try:
                stage_size = stage_data.batch["full_image"].shape[0]
                print(f"[DEBUG _restructure] Stage {stage_id}: {stage_size} envs", flush=True)
            except (KeyError, TypeError, AttributeError):
                print(f"[DEBUG _restructure] Stage {stage_id}: None envs", flush=True)

        return staged_data

    def _collate_trajectories(self, trajectories: dict, initial_state_ids: np.ndarray, meta_info) -> DataProto:
        """
        Collates the collected trajectory data into the final batch format.

        Combines data from all stages and all steps into a single batch.
        """
        # Flatten trajectories: combine all stages for each step
        flat_trajs = [{} for _ in range(len(trajectories[0]))]

        for stage_id in range(self.stage_num):
            for step_idx, step_data in enumerate(trajectories[stage_id]):
                if not flat_trajs[step_idx]:  # if dict is empty
                    flat_trajs[step_idx] = step_data
                else:
                    # Concatenate DataProto objects
                    for key, value in step_data.items():
                        if isinstance(value, DataProto):
                            flat_trajs[step_idx][key] = DataProto.concat([flat_trajs[step_idx][key], value])
                        elif isinstance(value, torch.Tensor):
                            flat_trajs[step_idx][key] = torch.cat([flat_trajs[step_idx][key], value], dim=0)

        # Extract and stack data from all steps
        all_pixel_values = [step["action"].batch["pixel_values"] for step in flat_trajs]
        all_responses = [step["action"].batch["responses"] for step in flat_trajs]
        all_input_ids = [step["action"].batch["input_ids"] for step in flat_trajs]
        all_attn_masks = [step["action"].batch["attention_mask"] for step in flat_trajs]
        all_actions = [step["action"].batch["action"] for step in flat_trajs]
        all_dones = [step["done"] for step in flat_trajs]

        pixel_values = torch.stack(all_pixel_values, dim=1)
        responses = torch.stack(all_responses, dim=1)
        input_ids = torch.stack(all_input_ids, dim=1)
        attention_mask = torch.stack(all_attn_masks, dim=1)
        actions = torch.stack(all_actions, dim=1)
        complete = torch.stack(all_dones, dim=1).squeeze(-1)  # Shape [bs, steps]

        batch_dict = {
            "pixel_values": pixel_values,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "complete": complete,
            "action": actions,
            "env_state_id": torch.from_numpy(initial_state_ids.astype(int)),
        }

        return DataProto.from_single_dict(batch_dict, meta_info=meta_info)
