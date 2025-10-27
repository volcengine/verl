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

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup


class EnvLoop:
    """An env loop manages interactions between models and vectorized environments. It's designed for computationally 
    intensive environments, such as robotics simulators."""

    def __init__(self, env_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, config: DictConfig):
        """
        Initialize the EnvLoop.

        Args:
            env_wg (RayWorkerGroup): Environment worker group.
            rollout_wg (RayWorkerGroup): Rollout worker group for model inference.
            config (DictConfig): YAML config.
        """
        self.env_wg = env_wg
        self.rollout_wg = rollout_wg
        self.config = config
        # Extract relevant configuration
        self.max_steps = config.env.train.max_episode_steps
        self.stage_num = config.rollout.pipeline_stage_num
        self.num_envs_per_worker = config.env.train.num_envs
        self.action_dim = config.actor.model.action_dim
        self.num_action_chunks = config.actor.model.num_action_chunks
        # Derived properties
        self.total_envs = self.env_wg.world_size * self.num_envs_per_worker
        if self.total_envs % self.stage_num != 0:
            raise ValueError(f"Total envs ({self.total_envs}) must be divisible by stage_num ({self.stage_num})")
        self.envs_per_stage = self.total_envs // self.stage_num

    async def run(self, prompts: DataProto) -> DataProto:
        """
        Run the environment interaction loop.
        This method orchestrates a pipelined process:
        1. Resets environments to specified initial states.
        2. In a loop, it gets actions from the rollout workers and applies them to the environments.
        3. Collects all trajectory data (observations, actions, rewards, dones).
        4. Formats and returns the collected trajectories as a single batch.
        Args:
            prompts (DataProto): Contains initial state IDs and other settings.
                                 - 'non_tensor_batch.state_ids': A numpy array of state IDs to reset envs.
        Returns:
            DataProto: A batch containing the complete trajectories.
        """
        initial_state_ids = prompts.non_tensor_batch["state_ids"]

        reset_prompts = DataProto.from_dict(non_tensors={"state_ids": initial_state_ids})
        reset_results = self.env_wg.reset_envs_to_state_ids(reset_prompts)

        staged_obs = self._restructure_obs_data(reset_results)
        # --- Pipeline state ---
        trajectories = {i: [] for i in range(self.stage_num)}  # To store (obs, action, rew, done) tuples
        rollout_futures = {}
        env_step_futures = {}
        is_complete = torch.zeros((self.total_envs,), dtype=torch.bool)

        for stage_id in range(self.stage_num):
            # trajectories[stage_id].append({'obs': staged_obs[stage_id]})
            trajectories[stage_id].append({})
            vla_input = staged_obs[stage_id]
            vla_input.meta_info = prompts.meta_info  # Pass along rollout config
            rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)
        for step in range(self.max_steps):
            if is_complete.all():
                print(f"All environments completed at step {step}. Ending early.")
                break

            for stage_id in range(self.stage_num):
                action_result: DataProto = rollout_futures[stage_id].get()

                trajectories[stage_id][-1]["action"] = action_result
                action_data = DataProto.from_dict(
                    non_tensors={"actions": action_result.batch["action"].cpu().numpy()},
                    meta_info={"stage_id": stage_id},
                )
                env_step_futures[stage_id] = self.env_wg.env_interact_step(action_data)

            staged_next_obs = {}
            for stage_id in range(self.stage_num):
                env_result: DataProto = env_step_futures[stage_id].get()

                # Store rewards and terminations
                trajectories[stage_id][-1]["rew"] = env_result.batch["rews"]
                trajectories[stage_id][-1]["done"] = env_result.batch["terminations"]

                # Update completion status
                stage_indices = slice(stage_id * self.envs_per_stage, (stage_id + 1) * self.envs_per_stage)
                is_complete[stage_indices] = is_complete[stage_indices] | trajectories[stage_id][-1]["done"].squeeze(-1)
                # Prepare next observation
                next_obs = DataProto(
                    batch=env_result.batch.select("full_image", "state"),
                    non_tensor_batch={"task_descriptions": env_result.non_tensor_batch["task_descriptions"]},
                )
                staged_next_obs[stage_id] = next_obs

            if step < self.max_steps - 1:
                for stage_id in range(self.stage_num):
                    # Start next trajectory step
                    # trajectories[stage_id].append({'obs': staged_next_obs[stage_id]})
                    trajectories[stage_id].append({})

                    # Send next observation to model
                    vla_input = staged_next_obs[stage_id]
                    vla_input.meta_info = prompts.meta_info  # Pass along rollout config
                    rollout_futures[stage_id] = self.rollout_wg.generate_sequences.remote(vla_input)
        self.env_wg.finish_rollout()
        return self._collate_trajectories(trajectories, initial_state_ids, meta_info=prompts.meta_info)

    def _restructure_obs_data(self, data_proto: DataProto) -> list[DataProto]:
        """Reshapes flat observation data from env_wg into a list of per-stage DataProto objects."""
        # env_wg returns a flat batch ordered by [worker0_stage0, worker0_stage1, ..., 
        # worker1_stage0, worker1_stage1, ...]
        # First, un-flatten by worker, then by stage

        num_workers = self.env_wg.world_size

        staged_data = [[] for _ in range(self.stage_num)]
        chunks = data_proto.chunk(num_workers)
        for worker_chunk in chunks:
            stage_chunks = worker_chunk.chunk(self.stage_num)
            for stage_id, data in enumerate(stage_chunks):
                staged_data[stage_id].append(data)

        # Concatenate data from all workers for each stage
        return [DataProto.concat(data_list) for data_list in staged_data]

    def _collate_trajectories(self, trajectories: dict, initial_state_ids: np.ndarray, meta_info) -> DataProto:
        """
        Collates the collected trajectory data into the final batch format.
        """
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

        all_pixel_values = [step["action"].batch["pixel_values"] for step in flat_trajs]
        all_responses = [step["action"].batch["responses"] for step in flat_trajs]
        all_input_ids = [step["action"].batch["input_ids"] for step in flat_trajs]
        all_attn_masks = [step["action"].batch["attention_mask"] for step in flat_trajs]
        all_actions = [step["action"].batch["action"] for step in flat_trajs]
        all_dones = [step["done"] for step in flat_trajs]
        # # Pad to max_steps if loop ended early
        # while len(all_actions) < self.max_steps:
        #      all_actions.append(torch.zeros_like(all_actions[0]))
        #      all_input_ids.append(torch.zeros_like(all_input_ids[0]))
        #      all_attn_masks.append(torch.zeros_like(all_attn_masks[0]))

        # while len(all_dones) < self.max_steps:
        #     all_dones.append(torch.ones_like(all_dones[0]))
        # Tensors need to have shape [bs, steps, ...]
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
            "env_state_id": torch.from_numpy(initial_state_ids),
        }

        return DataProto.from_single_dict(batch_dict, meta_info=meta_info)
