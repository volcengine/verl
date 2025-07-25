# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import logging
import threading
import time
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from recipe.fully_async_policy.RollouterActor import RollouterActor
from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type

import hydra
import ray

from recipe.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.dataset.rl_dataset import collate_fn

from fully_async_trainer import FullyAsyncTrainer

logger = logging.getLogger(__name__)


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class FullyAsyncTaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """运行完全异步的PPO训练"""
        setup_logging()

        logger.info("Starting fully async PPO training...")
        # 创建数据集和采样器
        logger.info("Creating dataset and sampler...")
        from verl.utils import hf_processor, hf_tokenizer

        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Define worker classes based on the actor strategy.
        if config.actor_rollout_ref.actor.strategy == "fsdp2":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup

            from recipe.one_step_off_policy.fsdp_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
                CriticWorker,
                RolloutWorker,
            )

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

            from recipe.one_step_off_policy.megatron_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
                CriticWorker,
                RolloutWorker,
            )

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from recipe.one_step_off_policy.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.Actor: ray.remote(actor_rollout_cls),
            Role.Rollout: ray.remote(RolloutWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "actor_pool"
        rollout_pool_id = "rollout_pool"

        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

        actor_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes

        resource_pool_spec = {
            "actor_pool": actor_pool,
            "rollout_pool": rollout_pool,
        }
        mapping = {
            Role.Actor: global_pool_id,
            Role.Rollout: rollout_pool_id,
            Role.Critic: global_pool_id,
        }
        print(f"resource_pool_spec: {resource_pool_spec}")
        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp2":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # 1. 创建MessageQueue
        logger.info("Creating MessageQueue...")
        # todo max_queue_size auto compute
        max_queue_size = config.async_training.get("max_queue_size", 1000)
        message_queue = MessageQueue.remote(config, max_queue_size)
        message_queue_client = MessageQueueClient(message_queue)

        # 2. 创建Rollouter Actor
        logger.info("Creating Rollouter...")
        rollouter_actor = RollouterActor.remote(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping={Role.Rollout: role_worker_mapping[Role.Rollout]},
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=RayWorkerGroup,
            processor=processor,
            train_dataset=train_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )

        # 初始化Rollouter
        ray.get(rollouter_actor.init_workers.remote())
        ray.get(rollouter_actor.set_message_queue_client.remote(message_queue_client))

        # 3. 创建Trainer
        logger.info("Creating FullyAsyncTrainer...")
        trainer_role_mapping = {
            role: worker_cls for role, worker_cls in role_worker_mapping.items() if role != Role.Rollout
        }

        # 创建奖励函数
        reward_fn, val_reward_fn = load_reward_manager(config, tokenizer)

        trainer = FullyAsyncTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=RayWorkerGroup,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )

        # 初始化Trainer
        trainer.init_workers()
        trainer.set_message_queue_client(message_queue_client)
        trainer.set_rollouter_actor(rollouter_actor)

        # 4. 设置参数同步
        logger.info("Setting up parameter synchronization...")
        # param_synchronizer = AsyncParameterSynchronizer(
        #     config=config, actor_wg=trainer.actor_wg, rollouter_actor=rollouter_actor
        # )

        # 5. 启动Rollouter（在后台线程中）
        logger.info("Starting Rollouter in background...")

        def run_rollouter():
            try:
                ray.get(rollouter_actor.fit.remote())
            except Exception as e:
                logger.error(f"Rollouter error: {e}")

        rollouter_thread = threading.Thread(target=run_rollouter, daemon=True)
        rollouter_thread.start()

        # 6. 启动Trainer（主线程）
        logger.info("Starting FullyAsyncTrainer...")
        trainer.fit()


@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    """主入口函数"""
    run_ppo(config, FullyAsyncTaskRunner)


if __name__ == "__main__":
    main()
