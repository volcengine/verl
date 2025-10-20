# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch

from verl.envs.action_utils import (
    put_info_on_image,
    save_rollout_video,
    tile_images,
    to_tensor,
)


class IsaacEnv(gym.Env):
    def __init__(self, cfg, rank, world_size):
        self.rank = rank
        self.cfg = cfg
        self.world_size = world_size
        self.seed = self.cfg.seed + rank
        self._is_start = True
        self.num_envs = self.cfg.num_envs

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)

        self.task_suite_name = self.cfg.task_suite_name

        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.max_episode_steps = cfg.max_episode_steps
        self.video_cfg = cfg.video_cfg

        self.render_images = []
        self.video_cnt = 0
        self.camera_name = cfg.init_params.camera_names

    def _init_env(self):
        """Initializes the Isaac Sim environment."""

        self.task_name = self.cfg.get("task_name")
        self.task_id = 0
        # FIXME since isaac use env to set task id, all env have to use the same task id
        if self.task_suite_name.startswith("libero"):
            os.environ["LIBERO_TASK_SUITE"] = self.task_suite_name
            if hasattr(self.cfg, "task_id") and self.cfg.task_id is not None:
                os.environ["LIBERO_TASK_ID"] = str(self.cfg.task_id)
                self.task_id = self.cfg.task_id
            else:
                os.environ["LIBERO_TASK_ID"] = "0"

            if not self.task_name:
                self.task_name = "Isaac-Libero-Franka-OscPose-v0"

            os.environ["LIBERO_OSC_TYPE"] = "pose_rel"

        # sys env must be set before import isaaclab
        from isaaclab.app import AppLauncher

        launch_args = {"headless": True, "enable_cameras": True}
        app_launcher = AppLauncher(**launch_args)
        self.app = app_launcher.app

        from isaaclab_tasks.utils import parse_env_cfg

        self.env_cfg = parse_env_cfg(self.task_name, num_envs=self.num_envs)
        self.env_cfg.env_name = self.cfg.get("env_name", str(self.task_id))
        # print(f"self.env_cfg: {self.env_cfg}")
        self.env_cfg.sim.device = self.cfg.get("device", "cuda")
        self.env_cfg.sim.physx.enable_ccd = True
        self.env_cfg.terminations.time_out = None
        self.env_cfg.observations.policy.concatenate_terms = False

        # create environment from loaded config
        self.env = gym.make(self.task_name, cfg=self.env_cfg).unwrapped

        if self.cfg.video_cfg.save_video:
            video_dir = os.path.join(self.cfg.video_cfg.video_base_dir, f"rank_{self.rank}")
            os.makedirs(video_dir, exist_ok=True)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print("Isaac Sim environment initialized.")
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")
        print(f"env_cfg.osc_type: {self.env_cfg.osc_type}")

        # TODO support other task suite
        if self.task_suite_name.startswith("libero"):
            self.task_descriptions = self.env.cfg.libero_config.task_info["language_instruction"]
        else:
            raise ValueError(f"Task suite {self.task_suite_name} is not supported.")
        print(f"libero_config.workspace_name: {self.env.cfg.libero_config.workspace_name}")

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        # Ensure terminations is a numpy array before the bitwise OR
        if isinstance(terminations, torch.Tensor):
            terminations = terminations.cpu().numpy()
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        if any(self.elapsed_steps > 0):
            episode_info["reward"] = episode_info["return"] / self.elapsed_steps
        else:
            episode_info["reward"] = 0
        infos["episode"] = to_tensor(episode_info)
        return infos

    def reset(self, env_idx: Optional[int | list[int] | np.ndarray] = None, options: Optional[dict] = None):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        # In Isaac Lab, reset is often for all envs, but we can reset metrics for a subset.
        if self.is_start:
            raw_obs, infos = self.env.reset()
        else:
            # This is tricky in isaaclab, as it auto-resets.
            # We will rely on the auto-reset from step.
            # For manual resets of some envs, we just reset their metrics.
            raw_obs = self.last_obs
            infos = self.last_infos
        obs = self._wrap_obs(raw_obs)

        self._reset_metrics(env_idx)

        # infos is already a dict from isaaclab, we just pass it on
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            obs, infos = self.reset()
            self._is_start = False
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)

            return (
                obs,
                torch.zeros(self.num_envs),
                to_tensor(terminations),
                to_tensor(truncations),
                infos,
            )
        truncations = self.elapsed_steps >= self.max_episode_steps
        # _actions = torch.zeros(self.action_space.shape)

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        self._elapsed_steps += 1
        # print(f"Step {self._elapsed_steps}: org actions {actions}, actual actions {_actions}")
        raw_obs, _reward, terminations, _, infos = self.env.step(actions)
        self.last_obs = raw_obs
        self.last_infos = infos

        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(_reward.cpu().numpy())

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": step_reward,
                "terminations": terminations,
                "task": self.task_descriptions,
            }
            self.add_new_frames(obs, plot_infos)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        # isaaclab handles auto-reset internally, so we don't need _handle_auto_reset
        # but we do need to reset metrics for envs that are done.
        dones = terminations.cpu().numpy() | truncations

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(actions, auto_reset=False)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(past_dones.cpu().numpy(), extracted_obs, infos)

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(env_idx=env_idx)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, reward):
        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            return reward

    def _wrap_obs(self, raw_obs):
        images_and_states = self._extract_image_and_state(raw_obs)

        obs = {
            "images_and_states": to_tensor(images_and_states),
            "task_descriptions": [self.task_descriptions] * self.num_envs,
        }
        return obs

    def _extract_image_and_state(self, obs):
        # TODO support multiple camera
        camera_name = self.camera_name[0]
        for key in self.env.unwrapped.scene.keys():
            if key.startswith(camera_name):
                cam = self.env.unwrapped.scene[key]
                break
        assert cam is not None, f"camera {camera_name} not found in scene"

        rgb = cam.data.output["rgb"]

        full_image = rgb.cpu().numpy()
        return {
            "full_image": full_image,
            "state": np.concatenate(
                [
                    obs["policy"]["eef_pose"].cpu(),
                    # quat2axisangle(obs["robot0_eef_quat"]), # isaac do not return robot0_eef_quat
                    # obs["policy"]["gripper_pos"].cpu(),
                ],
                axis=-1,
            ),
        }

    def add_new_frames(self, obs, plot_infos):
        images = []
        for env_id, img in enumerate(obs["images_and_states"]["full_image"]):
            info_item = {k: v if np.size(v) == 1 else v[env_id] for k, v in plot_infos.items()}
            img = put_info_on_image(img.cpu().numpy(), info_item)
            images.append(img)
        full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"rank_{self.rank}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
        )
        self.video_cnt += 1
        self.render_images = []

    def close(self):
        self.env.close()
        self.app.close()
        print("Isaac Sim environment closed.")

    def update_reset_state_ids(self):
        # TODO implement this method
        pass
