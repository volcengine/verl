# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The RLinf Authors.
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

import functools
import multiprocessing as mp
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.envs.libero_env.libero_env import LiberoEnv

# Monkey-patch torch.load to fix UnpicklingError with newer PyTorch versions.
# The libero library, a dependency for this environment, uses an older
# torch.load API. This is incompatible with the default 'weights_only=True'
# in PyTorch >= 2.6, which prevents loading files containing pickled Python objects.
# This patch restores the old default behavior (`weights_only=False`) for calls
# where the argument is not specified, allowing the environment to load its initial states.
original_torch_load = torch.load


@functools.wraps(original_torch_load)
def new_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)


torch.load = new_torch_load


if __name__ == "__main__":
    mp.set_start_method("spawn")  # solve CUDA compatibility problem

    # Basic configuration for the Libero environment
    cfg_dict = {
        "task_suite_name": "libero_10",
        "num_envs": 128,
        "group_size": 1,
        "num_group": 128,
        "seed": 0,
        "use_fixed_reset_state_ids": False,
        "ignore_terminations": False,
        "auto_reset": True,
        "max_episode_steps": 50,
        "use_rel_reward": False,
        "reward_coef": 1.0,
        "only_eval": False,
        "use_ordered_reset_state_ids": False,
        "num_images_in_input": 1,
        "init_params": {
            "camera_depths": False,
            "camera_heights": 256,
            "camera_widths": 256,
            "camera_names": ["agentview"],
        },
        "controller_configs": {
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2,
        },
        "video_cfg": {
            "save_video": True,
            "video_base_dir": "/tmp/videos",
        },
    }
    cfg = OmegaConf.create(cfg_dict)

    # Environment parameters
    n_envs = cfg.num_envs
    steps = 64
    action_dim = 7  # 6-DoF delta pose + 1 gripper
    times = 2

    # Initialize the environment
    libero_env = LiberoEnv(cfg, rank=0, world_size=1)

    for t in range(times):
        start_time = time.time()
        print(f"--- Episode {t + 1} ---")
        libero_env.is_start = True
        # The first call to step with actions=None will reset the environment
        obs_venv, _, terminated_venv, truncated_venv, info_venv = libero_env.step(actions=None)

        for step in range(steps):
            # Generate random actions
            actions = np.random.randn(n_envs, action_dim) * 0.5
            actions = np.clip(actions, -1, 1)

            # Step the environment
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = libero_env.step(actions)
            print(f"obs_venv : {obs_venv}")

            if terminated_venv.any() or truncated_venv.any():
                print(f"Step {step}:")
                if terminated_venv.any():
                    print(f"  Terminated: {terminated_venv.cpu().numpy()}")
                if truncated_venv.any():
                    print(f"  Truncated: {truncated_venv.cpu().numpy()}")
                if info_venv:
                    print(f"  Info: {info_venv}")

        if cfg.video_cfg.save_video:
            libero_env.flush_video(video_sub_dir=f"episode_{t}")

        end_time = time.time()
        print(f"--- Episode {t + 1} finished in {end_time - start_time:.2f} seconds ---")

    # Clean up the environment subprocesses
    libero_env.env.close()
    print("--- Test finished ---")
