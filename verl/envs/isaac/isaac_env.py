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

import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher

from verl.envs.action_utils import (
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

        self.task_name = cfg.task_name

        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg

    def _init_env(self):
        """Initializes the Isaac Sim environment."""
        launch_args = {"headless": self.cfg.headless}
        if self.cfg.video_cfg.save_video:
            launch_args["enable_cameras"] = True
        self.app_launcher = AppLauncher(**launch_args)
        self.sim = self.app_launcher.app

        from isaaclab_tasks.utils import parse_env_cfg

        self.env_cfg = parse_env_cfg(self.task_name, num_envs=self.num_envs)
        self.env_cfg.sim.device = self.cfg.device

        render_mode = "rgb_array" if self.cfg.video_cfg.save_video else None
        env = gym.make(self.task_name, cfg=self.env_cfg, render_mode=render_mode)

        if self.cfg.video_cfg.save_video:
            video_dir = os.path.join(self.cfg.video_cfg.video_base_dir, f"rank_{self.rank}")
            os.makedirs(video_dir, exist_ok=True)
            # Trigger recording for every episode, which starts on reset.
            self.env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda x: True,
                name_prefix="isaac-video",
            )
        else:
            self.env = env

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print("Isaac Sim environment initialized.")
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")

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

    def _wrap_obs(self, obs):
        return to_tensor(obs)

    def reset(self, env_idx: Optional[int | list[int] | np.ndarray] = None, options: Optional[dict] = None):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        # In Isaac Lab, reset is often for all envs, but we can reset metrics for a subset.
        if self.is_start:
            raw_obs, infos = self.env.reset()
            if self.cfg.video_cfg.save_video:
                pixels = self.env.render()
                if isinstance(raw_obs, dict):
                    raw_obs["pixels"] = pixels
        else:
            # This is tricky in isaaclab, as it auto-resets.
            # We will rely on the auto-reset from step.
            # For manual resets of some envs, we just reset their metrics.
            raw_obs = self.last_obs

        obs = self._wrap_obs(raw_obs)

        self._reset_metrics(env_idx)

        # infos is already a dict from isaaclab, we just pass it on
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            self.reset_video()
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

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu()
        else:
            actions = torch.from_numpy(actions).to(self.cfg.device)

        self._elapsed_steps += 1

        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        self.last_obs = raw_obs

        obs = self._wrap_obs(raw_obs)

        if self.cfg.video_cfg.save_video:
            pixels = self.env.render()
            if isinstance(obs, dict):
                obs["pixels"] = to_tensor(pixels)

        step_reward = self._calc_step_reward(_reward.cpu().numpy())

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        # isaaclab handles auto-reset internally, so we don't need _handle_auto_reset
        # but we do need to reset metrics for envs that are done.
        dones = terminations | truncations
        if dones.any():
            self._reset_metrics(np.where(dones.cpu())[0])

        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def _calc_step_reward(self, reward):
        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            return reward

    def reset_video(self):
        """Resets the video recording without saving the file."""
        if not self.video_cfg.save_video or not isinstance(self.env, gym.wrappers.RecordVideo):
            return

        if self.env.recording:
            self.env.recorded_frames = []
            self.env.recording = False
            self.env._video_name = None

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Saves the video of the current episode by closing the recorder."""
        if not self.video_cfg.save_video or not isinstance(self.env, gym.wrappers.RecordVideo):
            return

        if self.env.recording:
            # HACK: Access private member to get video path
            video_path = os.path.join(self.env.video_folder, f"{self.env._video_name}.mp4")

            self.env.stop_recording()

            if video_sub_dir:
                video_name = os.path.basename(video_path)
                new_video_dir = os.path.join(os.path.dirname(video_path), video_sub_dir)
                os.makedirs(new_video_dir, exist_ok=True)
                new_video_path = os.path.join(new_video_dir, video_name)
                try:
                    if os.path.exists(video_path):
                        os.rename(video_path, new_video_path)
                except OSError as e:
                    print(f"Error renaming video file: {e}")

    def close(self):
        self.env.close()
        self.sim.close()
        print("Isaac Sim environment closed.")
