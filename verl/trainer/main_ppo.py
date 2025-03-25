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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import os

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import ray
import hydra


@hydra.main(config_path='config', config_name='qwen7b_sft_async', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    import os, torch

    print("VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch sees device count:", torch.cuda.device_count())
    print("default device:", torch.cuda.current_device())
    print("default device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    if not ray.is_initialized():
        # this is for local ray cluster
        # ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN', 'VLLM_ATTENTION_BACKEND': 'XFORMERS', **dict(os.environ)}})

    main_task(config, compute_score)


def main_task(config, compute_score=None):
    from verl.utils.fs import copy_to_local
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    if config.trainer.get("hybrid_engine", True):
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
        }

        global_pool_id = 'global_pool'
        print(f'config.trainer.nnodes: {config.trainer.nnodes}')
        print(f'config.trainer.n_gpus_per_node: {config.trainer.n_gpus_per_node}')
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
    else:
        role_worker_mapping = {
            Role.Actor: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
            Role.Rollout: ray.remote(ActorRolloutRefWorker),
        }

        placement = config.trainer.placement
        resource_pool_spec = {
            "actor_pool": [placement["actor"]] * config.trainer.nnodes,
            "ref_pool": [placement["ref"]] * config.trainer.nnodes,
            "critic_pool": [placement.get("critic", 0)] * config.trainer.nnodes,
            "rollout_pool": [placement["rollout"]] * config.trainer.nnodes,
        }
        print(f"resource_pool_spec: {resource_pool_spec}")
        mapping = {
            Role.Actor: "actor_pool",
            Role.Critic: "critic_pool",
            Role.RefPolicy: "ref_pool",
            Role.Rollout: "rollout_pool",
        }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == "swedev":
        from verl.workers.reward_manager import SWEDevRewardManager
        reward_manager_cls = SWEDevRewardManager
    else:
        raise NotImplementedError
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            processor=processor,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
