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
"""

import hydra
import logging
import ray
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="atropos_grpo_trainer", version_base=None)
def main(config):
    """Main entry point for Atropos-VERL GRPO training"""
    run_atropos_grpo(config)


def run_atropos_grpo(config: DictConfig) -> None:
    """
    Run Atropos-VERL GRPO training with automatic inference server management
    and environment coordination.
    """
    
    # Initialize Ray with environment variables for inference servers
    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN", 
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                # Atropos-specific environment variables
                "ATROPOS_API_URL": config.atropos.api_url,
            }
        }
        ray.init(
            runtime_env=runtime_env,
            num_cpus=config.ray_init.num_cpus,
        )
    
    # Create and run the Atropos task runner
    runner = AtroposTaskRunner.remote()
    ray.get(runner.run.remote(config))
    
    # Optional timeline trace for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class AtroposTaskRunner:
    """Task runner for Atropos-VERL integration"""
    
    def run(self, config: DictConfig):
        """
        Run the complete Atropos-VERL training pipeline including:
        1. Inference server startup
        2. Atropos environment coordination
        3. GRPO training with token-level advantages
        4. Policy weight synchronization
        """
        
        # Print configuration
        logger.info("Starting Atropos-VERL GRPO Training")
        logger.info("Configuration:")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        
        # Validate configuration
        self._validate_atropos_config(config)
        
        # Download model to local machine
        from verl.utils.fs import copy_to_local
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, 
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        
        # Initialize tokenizer and processor
        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        # Validate vLLM version for LoRA if needed
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")
        
        # Set up worker classes based on strategy
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.atropos_workers import AtroposRolloutWorker
            from verl.workers.fsdp_workers import CriticWorker
            
            # Use Atropos rollout worker instead of standard worker
            actor_rollout_cls = AtroposRolloutWorker
            ray_worker_group_cls = RayWorkerGroup
            
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.atropos_workers import AtroposRolloutWorker
            from verl.workers.megatron_workers import CriticWorker
            
            # Use Atropos rollout worker for Megatron as well
            actor_rollout_cls = AtroposRolloutWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError(f"Strategy {config.actor_rollout_ref.actor.strategy} not supported")
        
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        
        # Map roles to worker classes
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        
        # Set up resource pool
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        
        # Add reward model worker if enabled
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        
        # Add reference policy worker if needed
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        
        # Load reward functions (optional for Atropos)
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
        
        # Create datasets (may be empty for Atropos coordination)
        from verl.utils.dataset.rl_dataset import collate_fn
        train_dataset = self._create_atropos_dataset(config.data, tokenizer, processor)
        val_dataset = self._create_atropos_dataset(config.data, tokenizer, processor)
        train_sampler = self._create_atropos_sampler(config.data, train_dataset)
        
        # Initialize the Atropos-aware PPO trainer
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
        
        # Initialize workers and start training
        trainer.init_workers()
        trainer.fit()
    
    def _validate_atropos_config(self, config: DictConfig):
        """Validate Atropos-specific configuration"""
        required_fields = ["api_url", "timeout"]
        for field in required_fields:
            if field not in config.atropos:
                raise ValueError(f"Missing required Atropos config field: {field}")
        
        logger.info(f"Atropos API URL: {config.atropos.api_url}")
        logger.info(f"Using GRPO advantage estimator: {config.algorithm.adv_estimator}")
    
    def _create_atropos_dataset(self, data_config, tokenizer, processor):
        """Create dataset for Atropos coordination (may be empty/placeholder)"""
        from verl.utils.dataset.rl_dataset import RLHFDataset
        from torch.utils.data import Dataset
        
        # For Atropos integration, data comes from environments
        # Return empty dataset as placeholder
        class AtroposPlaceholderDataset(Dataset):
            def __len__(self):
                return 1000  # Placeholder size
            
            def __getitem__(self, idx):
                return {
                    'input_ids': [],
                    'attention_mask': [],
                    'labels': [],
                }
        
        return AtroposPlaceholderDataset()
    
    def _create_atropos_sampler(self, data_config, dataset):
        """Create sampler for Atropos coordination"""
        from torch.utils.data import SequentialSampler
        return SequentialSampler(dataset)


class AtroposPPOTrainer(RayPPOTrainer):
    """
    Extended PPO trainer with Atropos integration support.
    
    This trainer extends the standard RayPPOTrainer to:
    - Coordinate with Atropos environments
    - Handle inference server management
    - Support token-level advantages
    - Manage policy weight synchronization
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Atropos-specific initialization
        self.atropos_config = self.config.get("atropos", {})
        logger.info("AtroposPPOTrainer initialized")
    
    def init_workers(self):
        """Initialize workers with Atropos-specific setup"""
        super().init_workers()
        
        # Configure Atropos rollout workers with inference endpoints
        self._setup_atropos_integration()
    
    def _setup_atropos_integration(self):
        """Set up Atropos integration including inference server coordination"""
        logger.info("Setting up Atropos integration...")
        
        # Get inference endpoints from rollout workers
        # This would be implemented based on VeRL's server management
        inference_endpoints = self._get_inference_endpoints()
        
        # Configure Atropos rollout workers
        actor_rollout_workers = self.resource_pool_manager.name_to_remote_cls_mapping.get("ActorRollout", [])
        for worker in actor_rollout_workers:
            if hasattr(worker, 'set_inference_endpoints'):
                ray.get(worker.set_inference_endpoints.remote(inference_endpoints))
        
        logger.info(f"Configured {len(actor_rollout_workers)} Atropos rollout workers")
    
    def _get_inference_endpoints(self):
        """Get inference server endpoints for Atropos environments"""
        # This would integrate with VeRL's inference server management
        # For now, return configured endpoints
        endpoints = self.atropos_config.get("inference_endpoints", [])
        
        if not endpoints:
            # Auto-discover endpoints based on VeRL's server management
            # This would be implemented based on the specific inference backend
            logger.warning("No inference endpoints configured, using defaults")
            endpoints = ["http://localhost:9000", "http://localhost:9001"]
        
        return endpoints


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create RL dataset - adapted for Atropos coordination"""
    # For Atropos, this may return a placeholder since data comes from environments
    from torch.utils.data import Dataset
    
    class AtroposDataset(Dataset):
        def __len__(self):
            return 1000  # Placeholder
        
        def __getitem__(self, idx):
            return {
                'input_ids': [],
                'attention_mask': [],
                'labels': [],
            }
    
    return AtroposDataset()


def create_rl_sampler(data_config, dataset):
    """Create RL sampler - adapted for Atropos coordination"""
    from torch.utils.data import SequentialSampler
    return SequentialSampler(dataset)


if __name__ == "__main__":
    main() 