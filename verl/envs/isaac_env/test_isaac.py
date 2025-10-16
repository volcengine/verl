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
    actions = np.array(
        [
            [5.59112417e-01, 8.06460073e-02, 1.36817226e-02, -4.64279854e-04, -1.72158767e-02, -6.57548380e-04, 0],
            [2.12711899e-03, -3.13366604e-01, 3.41386353e-04, -4.64279854e-04, -8.76528812e-03, -6.57548380e-04, 0],
            [7.38182960e-02, -4.64548351e-02, -6.63602950e-02, -4.64279854e-04, -2.32520114e-02, -6.57548380e-04, 0],
            [7.38182960e-02, -1.60845593e-01, 3.41386353e-04, -4.64279854e-04, 1.05503430e-02, -6.57548380e-04, 0],
            [7.38182960e-02, -3.95982152e-01, -7.97006313e-02, -5.10713711e-03, 3.22804279e-02, -6.57548380e-04, 0],
            [2.41859427e-02, -3.64206941e-01, -6.63602950e-02, -4.64279854e-04, 1.05503430e-02, -6.57548380e-04, 0],
            [4.62447664e-02, -5.16727952e-01, -7.97006313e-02, -4.64279854e-04, 1.05503430e-02, 8.73740975e-03, 0],
            [4.62447664e-02, -5.73923331e-01, 3.41386353e-04, -4.64279854e-04, 6.92866212e-03, -6.57548380e-04, 0],
            [1.56538885e-01, 8.05120809e-01, 3.41386353e-04, -4.64279854e-04, -3.14699528e-04, -6.57548380e-04, 0],
            [1.23450649e-01, -6.24763668e-01, -4.63497906e-02, 1.39286305e-03, -3.14699528e-04, -6.57548380e-04, 0],
            [5.97715359e-01, -5.99343500e-01, -5.30199588e-02, 1.39286305e-03, -3.14699528e-04, -6.57548380e-04, 0],
            [6.80435947e-01, -5.48503163e-01, -6.32878179e-03, -4.64279854e-04, -3.14699528e-04, -6.57548380e-04, 0],
            [5.48083005e-01, -5.48503163e-01, -6.32878179e-03, -4.64279854e-04, 1.41720238e-02, -2.41449437e-02, 0],
            [5.37053593e-01, -6.24763668e-01, -5.30199588e-02, -4.64279854e-04, 1.41720238e-02, -2.17962042e-02, 0],
            [6.80435947e-01, -6.24763668e-01, -5.30199588e-02, 7.89286320e-03, -3.14699528e-04, 4.63172423e-02, 0],
            [7.64182492e-03, -6.05698542e-01, -5.30199588e-02, 7.89286320e-03, -2.72915341e-03, 2.04811074e-02, 0],
            [3.55068298e-01, 6.20824588e-01, 1.07064077e-01, 1.06785776e-02, -3.14699528e-04, 4.03993069e-03, 0],
            [6.80435947e-01, -3.38786773e-01, -6.63602950e-02, 5.10714885e-03, -3.14699528e-04, 6.38867022e-03, 0],
            [6.80435947e-01, -1.22715341e-01, -8.63707995e-02, -4.64279854e-04, -3.14699528e-04, -6.57548380e-04, 0],
            [-3.38758693e-03, -1.67200635e-01, -2.79805676e-01, -4.64279854e-04, -2.56664653e-02, -6.57548380e-04, 1],
            [-3.38758693e-03, -2.49816183e-01, -2.79805676e-01, -4.64279854e-04, -3.14699528e-04, -6.57548380e-04, 1],
            [4.07300604e-02, -2.49816183e-01, -2.86475844e-01, -4.64279854e-04, -3.14699528e-04, 4.03993069e-03, 1],
            [5.72741782e-02, -2.11685930e-01, -2.86475844e-01, -4.64279854e-04, -1.72158767e-02, -2.88424228e-02, 1],
            [3.05435945e-01, -2.43461141e-01, -2.86475844e-01, -4.64279854e-04, -3.14699528e-04, -2.88424228e-02, 1],
        ]
    )

    # mp.set_start_method("spawn")
    num_envs = 1
    # Basic configuration for the Isaac environment
    cfg = OmegaConf.create(
        {
            "task_name": "Isaac-Libero-Franka-OscPose-v0",
            "task_suite_name": "libero_10",
            "task_id": 0,
            "num_envs": num_envs,
            "device": "cuda:0",
            "seed": 42,
            "max_episode_steps": 512,
            "video_cfg": {
                "save_video": True,
                "video_base_dir": "/tmp/videos",
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

    # Initialize the environment
    isaac_env = IsaacEnv(cfg, rank=0, world_size=1)
    # The first call to step with actions=None will reset the environment
    obs_venv, _, terminated_venv, truncated_venv, info_venv = isaac_env.step(actions=None)

    isaac_env.is_start = True

    step = 0
    for action in actions:
        # Generate random actions
        # Step the environment
        # For CartPole, action dimension is 1
        # actions = torch.rand(isaac_env.num_envs, 1, device=cfg.device)
        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = isaac_env.step(
            np.array([action] * num_envs)
        )
        print(f"Step {step}: input: {action} Observation: {obs_venv['images_and_states']['state']}")

        if terminated_venv.any() or truncated_venv.any():
            print(f"Step {step}:")
            if terminated_venv.any():
                print(f"  Terminated: {terminated_venv.cpu().numpy()}")
            if truncated_venv.any():
                print(f"  Truncated: {truncated_venv.cpu().numpy()}")
            if info_venv:
                print(f"  Info: {info_venv}")
            break
        step += 1
    end_time = time.time()

    if cfg.video_cfg.save_video:
        isaac_env.flush_video(video_sub_dir=f"episode_{0}")

    print("--- Test finished ---")
    # Clean up the environment
    isaac_env.close()
