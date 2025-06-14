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
Main Atropos-VERL GRPO Trainer

This module provides the main training entry point for Atropos-VERL integration,
implementing GRPO training with multi-environment coordination through the Atropos API.

Key features:
- Automatic inference server management (vLLM/SGLang)
- Atropos API integration for environment coordination
- Token-level advantage support from Atropos environments
- Full online RL with policy weight synchronization
- GPU device management and memory optimization
"""

import hydra
import logging
import os
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, SequentialSampler

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.dataset.rl_dataset import collate_fn


logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="atropos_grpo_trainer", version_base=None)
def main(config):
    """Main entry point for Atropos-VERL GRPO training"""
    run_atropos_grpo(config)


def run_atropos_grpo(config: DictConfig) -> None:
    
    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
        torch.cuda.init()
        logger.info(f"CUDA setup: {torch.cuda.device_count()} GPUs")
    
    if not ray.is_initialized():
        ray_kwargs = {
            "runtime_env": {"env_vars": {"ATROPOS_API_URL": config.atropos.api_url}},
            "num_cpus": config.ray_init.num_cpus,
        }
        if torch.cuda.is_available():
            ray_kwargs["num_gpus"] = torch.cuda.device_count()
        ray.init(**ray_kwargs)
    
    runner = AtroposTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class AtroposTaskRunner:
    def run(self, config: DictConfig):
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        
        OmegaConf.resolve(config)
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, 
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        # Setup workers based on strategy
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.atropos_workers import AtroposRolloutWorker
            from verl.workers.fsdp_workers import CriticWorker
            actor_rollout_cls, ray_worker_group_cls = AtroposRolloutWorker, RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.atropos_workers import AtroposRolloutWorker
            from verl.workers.megatron_workers import CriticWorker
            actor_rollout_cls, ray_worker_group_cls = AtroposRolloutWorker, NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")
        
        gpu_resources = 1 if torch.cuda.is_available() else 0
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(num_gpus=gpu_resources)(actor_rollout_cls),
            Role.Critic: ray.remote(num_gpus=gpu_resources)(CriticWorker),
        }
        
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRollout: global_pool_id, Role.Critic: global_pool_id}
        
        # Add optional workers
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(num_gpus=gpu_resources)(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker
            role_worker_mapping[Role.RefPolicy] = ray.remote(num_gpus=gpu_resources)(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        
        # Load reward functions
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, 
            **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1,
            **config.reward_model.get("reward_kwargs", {})
        )
        
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, 
            mapping=mapping
        )
        
        # Create datasets
        class AtroposDataset(Dataset):
            def __len__(self): return 1000
            def __getitem__(self, idx): return {'input_ids': [], 'attention_mask': [], 'labels': []}
        
        train_dataset = val_dataset = AtroposDataset()
        train_sampler = SequentialSampler(train_dataset)
        
        # Initialize trainer
        trainer = AtroposPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        
        trainer.init_workers()
        trainer.fit()


class AtroposPPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atropos_config = self.config.get("atropos", {})
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    
    def init_workers(self):
        super().init_workers()
        endpoints = self.atropos_config.get("inference_endpoints", 
                    [f"http://localhost:{9000 + i}" for i in range(min(torch.cuda.device_count(), 4))] 
                    if torch.cuda.is_available() else ["http://localhost:9000"])
        
        actor_rollout_workers = self.resource_pool_manager.name_to_remote_cls_mapping.get("ActorRollout", [])
        for worker in actor_rollout_workers:
            if hasattr(worker, 'set_inference_endpoints'):
                ray.get(worker.set_inference_endpoints.remote(endpoints))


if __name__ == "__main__":
    main() 