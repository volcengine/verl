# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


import datasets
import hydra
import ray
from omegaconf import OmegaConf
import torch

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl import DataProto


# from verl.trainer.ppo.ray_trainer import RayTrainer
from .rob_ray_trainer import RobRayPPOTrainer

def calculate_reward(data: DataProto, return_dict: bool = False) -> torch.Tensor:
    complete_tensor = data.batch["complete"]
    batch_size, num_steps = complete_tensor.shape[:2]
    traj_has_complete = torch.any(complete_tensor, dim=(1, 2))  # shape: [batch_size]
    reward_per_traj = traj_has_complete.float()
    reward_per_step = reward_per_traj.unsqueeze(1).expand(batch_size, num_steps)
    if return_dict:
        return {"reward_tensor": reward_per_step}
    else:
        return reward_per_step

@hydra.main(config_path="config", config_name="rob_ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.utils.fs import copy_local_path_from_hdfs

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from recipe.vla.workers.env.env_worker import EnvWorker
        from verl.single_controller.ray import RayWorkerGroup

        from .fsdp_workers import RobActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        # Role.Critic: ray.remote(RobActorRolloutRefWorker),
        Role.ActorRollout: ray.remote(RobActorRolloutRefWorker),
        # Role.RefPolicy: ray.remote(RobActorRolloutRefWorker),
        Role.Env: ray.remote(EnvWorker),
    }

    global_pool_id = "global_pool"
    env_gpu_num = int(config.trainer.n_gpus_per_node * 2 / 3)
    rollout_gpu_num = config.trainer.n_gpus_per_node - env_gpu_num
    resource_pool_spec = {
        global_pool_id: [env_gpu_num] * config.trainer.nnodes,
        "env_gpu_pool": [rollout_gpu_num] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        # Role.Critic: global_pool_id,
        # Role.RefPolicy: global_pool_id,
        Role.Env: "env_gpu_pool",
    }

    # reward_fn = RobRewardManager( num_examine=0, config=config) # note: verifier is called
    # both inside reward_fn and outside.

    # # Note that we always use function-based RM for validation
    # val_reward_fn = RobRewardManager( num_examine=1,config=config)
    reward_fn = calculate_reward
    val_reward_fn = calculate_reward

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # Create training and validation datasets.
    train_dataset = datasets.load_dataset("parquet", data_files=config.data.train_files)["train"]
    val_dataset = datasets.load_dataset("parquet", data_files=config.data.val_files)["train"]

    trainer = RobRayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
