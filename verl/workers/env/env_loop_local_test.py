# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import ray
from omegaconf import OmegaConf

from verl.workers.env.env_worker import EnvWorker
from verl.workers.rollout.naive.naive_rollout_rob import NaiveRolloutRob
from verl.workers.config.rollout import RolloutConfig

if not ray.is_initialized():
    ray.init()

RayEnvWorker = ray.remote(num_gpus=1)(EnvWorker)
stage_num = 2
cfg_dict = {
    "rollout": {"pipeline_stage_num": stage_num},
    "env": {
        "train": {
            "use_fixed_reset_state_ids": False,
            "ignore_terminations": False,
            "auto_reset": True,
            "max_episode_steps": 512,
            "use_rel_reward": False,
            "reward_coef": 1.0,
            "only_eval": False,
            "use_ordered_reset_state_ids": False,
            # "num_images_in_input": 1,
            "init_params": {
                "camera_depths": False,
                "camera_heights": 256,
                "camera_widths": 256,
                "camera_names": ["agentview", "robot0_eye_in_hand"],
            },
            "video_cfg": {
                "save_video": True,
                "video_base_dir": "/tmp/videos",
            },
            "task_suite_name": "libero_10",
            "num_envs": 16,
            "num_group": 2,
            "group_size": 8,
            "simulator_type": "libero",
            "seed": 0,
        },
        "enable_offload": False,
    },
    "actor": {"model": {"num_action_chunks": 8, "action_dim": 7}},
    "runner": {"only_eval": False},
}
env_cfg = OmegaConf.create(cfg_dict)
env_worker = RayEnvWorker.remote(env_cfg, 0, 1)

init_out = env_worker.init_worker.remote()
ray.get(init_out)

ref = env_worker._init_simulator.remote()
last_obs_list, last_dones_list = ray.get(ref)

RayNaiveRolloutRob = ray.remote(num_gpus=1)(NaiveRolloutRob)
rollout_config = RolloutConfig(do_sample=True, temperature=1.6, prompt_length=512)
model_config = {"local_path": "/file_system/common-models/Haozhan72-kangsheng/Openvla-oft-SFT-libero10-trajall"}
rollout_workers = RayNaiveRolloutRob.remote(rollout_config, model_config, device_mesh=None)

env_obs_refs = {}
rollout_refs = {}
for _ in range(512 // 8):
    for stage_id in range(stage_num):
        if _ == 0:
            rollout_refs[stage_id] = rollout_workers.generate_sequences.remote(last_obs_list[stage_id])
        else:
            env_batch = ray.get(env_obs_refs[stage_id])
            obs = env_batch["obs"]
            rollout_refs[stage_id] = rollout_workers.generate_sequences.remote(obs)
    for stage_id in range(stage_num):
        batch = ray.get(rollout_refs[stage_id])
        action = batch["actions"]
        action = action.cpu().numpy()
        # already in env
        # normalized_action = normalize_gripper_action(action, binarize=True)
        # inverted_action = invert_gripper_action(normalized_action)
        env_obs_refs[stage_id] = env_worker.env_interact_step.remote(action, stage_id=stage_id)


finish_ref = env_worker.finish_rollout.remote()
ray.get(finish_ref)

ray.timeline(filename="2stage_pipeline_timeline.json")
