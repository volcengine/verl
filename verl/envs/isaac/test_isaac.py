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

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.envs.isaac.isaac_env import IsaacEnv

if __name__ == "__main__":
    # This is needed for Isaac Sim
    # mp.set_start_method("spawn")

    # Basic configuration for the Isaac environment
    cfg = OmegaConf.create(
        {
            "task_name": "Isaac-Cartpole-v0",
            "num_envs": 4,
            "device": "cuda:0",
            "headless": True,
            "seed": 42,
            "video_cfg": {
                "save_video": True,
                "video_base_dir": "/tmp/videos",
                "video_length": 200,
                "video_interval": 2000,
            },
            "ignore_terminations": False,
            "auto_reset": True,
            "use_rel_reward": False,
        }
    )

    # Environment parameters
    n_envs = cfg.num_envs
    steps = 50
    # This should match the environment's action space
    action_dim = 8
    times = 2

    # Initialize the environment
    isaac_env = IsaacEnv(cfg, rank=0, world_size=1)

    for t in range(times):
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
            actions = torch.rand(isaac_env.num_envs, 1, device=cfg.device)
            for i in range(10):
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = isaac_env.step(actions)
                print("obs keys:", obs_venv.keys())
                if "pixels" in obs_venv:
                    print("pixels shape:", obs_venv["pixels"].shape)
                    print("pixels dtype:", obs_venv["pixels"].dtype)
                else:
                    print("no pixels in obs {obs_venv}")

            print(
                f"obs_venv {obs_venv}\n, reward_venv {reward_venv}\n, \
                terminated_venv {terminated_venv}\n, truncated_venv \
                {truncated_venv}\n, info_venv {info_venv}\n"
            )
            if terminated_venv.any() or truncated_venv.any():
                print(f"Step {step}:")
                if terminated_venv.any():
                    print(f"  Terminated: {terminated_venv.cpu().numpy()}")
                if truncated_venv.any():
                    print(f"  Truncated: {truncated_venv.cpu().numpy()}")
                if info_venv:
                    print(f"  Info: {info_venv}")

        if cfg.video_cfg.save_video:
            isaac_env.flush_video(video_sub_dir=f"episode_{t}")

    # Clean up the environment
    isaac_env.close()
    print("--- Test finished ---")
