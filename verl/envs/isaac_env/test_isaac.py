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

import time

import numpy as np
from omegaconf import OmegaConf

from verl.envs.isaac_env.isaac_env import IsaacEnv

if __name__ == "__main__":
    # This is needed for Isaac Sim
    # mp.set_start_method("spawn")

    # Basic configuration for the Isaac environment
    cfg = OmegaConf.create(
        {
            "task_name": "Isaac-Libero-Franka-Replay-Camera-v0",
            "task_suite": "libero_10",
            "num_envs": 4,
            "device": "cuda:0",
            "headless": True,
            "seed": 42,
            "max_episode_steps": 8,
            "video_cfg": {
                "save_video": True,
                "video_base_dir": "/tmp/videos",
                "video_length": 200,
                "video_interval": 2000,
            },
            "ignore_terminations": False,
            "auto_reset": True,
            "use_rel_reward": False,
            "init_params": {
                "camera_names": ["agentview"],
            },
            # "controller_configs": {
            #     "type": "OSC_POSE",
            #     "input_max": 1,
            #     "input_min": -1,
            #     "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            #     "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            #     "kp": 150,
            #     "damping_ratio": 1,
            #     "impedance_mode": "fixed",
            #     "kp_limits": [0, 300],
            #     "damping_ratio_limits": [0, 10],
            #     "position_limits": None,
            #     "orientation_limits": None,
            #     "uncouple_pos_ori": True,
            #     "control_delta": True,
            #     "interpolation": None,
            #     "ramp_ratio": 0.2,
            # },
        }
    )

    # Environment parameters
    n_envs = cfg.num_envs
    steps = 512
    # This should match the environment's action space
    action_dim = 8
    times = 2

    # Initialize the environment
    isaac_env = IsaacEnv(cfg, rank=0, world_size=1)

    for t in range(times):
        start_time = time.time()
        print(f"--- Episode {t + 1} ---")
        isaac_env.is_start = True
        # The first call to step with actions=None will reset the environment
        obs_venv, _, terminated_venv, truncated_venv, info_venv = isaac_env.step(actions=None)
        for step in range(steps):
            # Generate random actions
            actions = np.random.randn(n_envs, action_dim) * 0.5
            actions = np.clip(actions, -1, 1)

            # Step the environment
            # For CartPole, action dimension is 1
            # actions = torch.rand(isaac_env.num_envs, 1, device=cfg.device)
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = isaac_env.step(actions)
            if terminated_venv.any() or truncated_venv.any():
                print(f"Step {step}:")
                if terminated_venv.any():
                    print(f"  Terminated: {terminated_venv.cpu().numpy()}")
                if truncated_venv.any():
                    print(f"  Truncated: {truncated_venv.cpu().numpy()}")
                if info_venv:
                    print(f"  Info: {info_venv}")
                break
        end_time = time.time()
        print(f"--- Episode {t + 1} finished in {end_time - start_time:.2f} seconds ---")

        if cfg.video_cfg.save_video:
            isaac_env.flush_video(video_sub_dir=f"episode_{t}")
    print("--- Test finished ---")
    # Clean up the environment
    isaac_env.close()
