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
    Run Atropos-VERL GRPO training with automatic inference server management,
    environment coordination, and GPU optimization.
    """
    
    # Initialize CUDA environment variables for distributed training
    _setup_cuda_environment()
    
    # Initialize Ray with GPU-optimized runtime environment
    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN", 
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                # GPU optimization environment variables
                "CUDA_LAUNCH_BLOCKING": "0",  # Enable async CUDA operations
                "TORCH_CUDNN_DETERMINISTIC": "0",  # Allow non-deterministic ops for speed
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",  # Optimize memory allocation
                # Atropos-specific environment variables
                "ATROPOS_API_URL": config.atropos.api_url,
            }
        }
        
        # Configure Ray for GPU-optimized operation
        ray_init_kwargs = {
            "runtime_env": runtime_env,
            "num_cpus": config.ray_init.num_cpus,
        }
        
        # Add GPU configuration if available
        if torch.cuda.is_available():
            ray_init_kwargs["num_gpus"] = torch.cuda.device_count()
            logger.info(f"Initializing Ray with {torch.cuda.device_count()} GPUs")
        
        ray.init(**ray_init_kwargs)
    
    # Create and run the Atropos task runner
    runner = AtroposTaskRunner.remote()
    ray.get(runner.run.remote(config))
    
    # Optional timeline trace for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


def _setup_cuda_environment():
    """Setup CUDA environment variables for optimal GPU performance"""
    if torch.cuda.is_available():
        # Optimize CUDA memory allocation
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
        
        # Enable CUDA caching allocator
        os.environ.setdefault("CUDA_CACHE_DISABLE", "0")
        
        # Set optimal CUDA stream settings
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        
        # Initialize CUDA context early
        torch.cuda.init()
        
        logger.info(f"CUDA setup complete. Available devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    else:
        logger.warning("CUDA not available - using CPU only")


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class AtroposTaskRunner:
    """GPU-optimized task runner for Atropos-VERL integration"""
    
    def run(self, config: DictConfig):
        """
        Run the complete Atropos-VERL training pipeline including:
        1. GPU initialization and optimization
        2. Inference server startup with GPU allocation
        3. Atropos environment coordination
        4. GRPO training with token-level advantages
        5. Policy weight synchronization with GPU memory management
        """
        
        # Initialize GPU context in the worker
        self._init_gpu_context()
        
        # Print configuration
        logger.info("Starting Atropos-VERL GRPO Training")
        logger.info("Configuration:")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        
        # Validate configuration
        self._validate_atropos_config(config)
        
        # Download model to local machine with GPU considerations
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
        
        # Set up worker classes based on strategy with GPU support
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
        
        # Map roles to worker classes with GPU resource allocation
        gpu_resources = 1 if torch.cuda.is_available() else 0
        
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(num_gpus=gpu_resources)(actor_rollout_cls),
            Role.Critic: ray.remote(num_gpus=gpu_resources)(CriticWorker),
        }
        
        # Set up resource pool with GPU allocation
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
            role_worker_mapping[Role.RewardModel] = ray.remote(num_gpus=gpu_resources)(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        
        # Add reference policy worker if needed
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker
            role_worker_mapping[Role.RefPolicy] = ray.remote(num_gpus=gpu_resources)(ActorRolloutRefWorker)
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
        
        # Initialize the Atropos-aware PPO trainer with GPU optimization
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
    
    def _init_gpu_context(self):
        """Initialize GPU context in the Ray worker"""
        if torch.cuda.is_available():
            # Initialize CUDA in worker
            torch.cuda.init()
            device_count = torch.cuda.device_count()
            
            # Set device based on Ray worker GPU allocation
            if device_count > 0:
                # Ray automatically sets CUDA_VISIBLE_DEVICES for workers
                # Use device 0 as it will be the only visible device
                torch.cuda.set_device(0)
                logger.info(f"Worker initialized with GPU: {torch.cuda.current_device()}")
                
                # Warm up GPU
                dummy_tensor = torch.zeros(1, device='cuda')
                del dummy_tensor
                torch.cuda.empty_cache()
            else:
                logger.warning("No CUDA devices available in worker")
        else:
            logger.warning("CUDA not available in worker")
    
    def _validate_atropos_config(self, config: DictConfig):
        """Validate Atropos-specific configuration"""
        required_fields = ["api_url"]
        
        # Add default timeout if not specified
        if "timeout" not in config.atropos:
            config.atropos.timeout = 30
        
        for field in required_fields:
            if field not in config.atropos:
                raise ValueError(f"Missing required Atropos config field: {field}")
        
        logger.info(f"Atropos API URL: {config.atropos.api_url}")
        logger.info(f"Using GRPO advantage estimator: {config.algorithm.adv_estimator}")
        
        # Validate GPU configuration
        if torch.cuda.is_available():
            logger.info(f"GPU acceleration enabled with {torch.cuda.device_count()} devices")
        else:
            logger.warning("GPU acceleration not available - training will use CPU")
    
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
    Extended PPO trainer with Atropos integration and GPU optimization support.
    
    This trainer extends the standard RayPPOTrainer to:
    - Coordinate with Atropos environments
    - Handle inference server management with GPU allocation
    - Support token-level advantages
    - Manage policy weight synchronization with GPU memory optimization
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Atropos-specific initialization
        self.atropos_config = self.config.get("atropos", {})
        
        # GPU context initialization
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            logger.info(f"AtroposPPOTrainer using GPU: {self.device}")
        else:
            self.device = torch.device("cpu")
            logger.warning("AtroposPPOTrainer using CPU")
        
        logger.info("AtroposPPOTrainer initialized with GPU optimization")
    
    def init_workers(self):
        """Initialize workers with Atropos-specific setup and GPU allocation"""
        super().init_workers()
        
        # Configure Atropos rollout workers with inference endpoints
        self._setup_atropos_integration()
    
    def _setup_atropos_integration(self):
        """Set up Atropos integration including GPU-optimized inference server coordination"""
        logger.info("Setting up Atropos integration with GPU optimization...")
        
        # Get inference endpoints from rollout workers
        # This would be implemented based on VeRL's server management
        inference_endpoints = self._get_inference_endpoints()
        
        # Configure Atropos rollout workers with GPU information
        actor_rollout_workers = self.resource_pool_manager.name_to_remote_cls_mapping.get("ActorRollout", [])
        for worker in actor_rollout_workers:
            if hasattr(worker, 'set_inference_endpoints'):
                ray.get(worker.set_inference_endpoints.remote(inference_endpoints))
        
        logger.info(f"Configured {len(actor_rollout_workers)} Atropos rollout workers with GPU support")
    
    def _get_inference_endpoints(self):
        """Get GPU-optimized inference server endpoints for Atropos environments"""
        # This would integrate with VeRL's inference server management
        # For now, return configured endpoints
        endpoints = self.atropos_config.get("inference_endpoints", [])
        
        if not endpoints:
            # Auto-discover endpoints based on VeRL's server management
            # This would be implemented based on the specific inference backend
            logger.warning("No inference endpoints configured, using defaults")
            # Generate endpoints based on available GPUs
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                endpoints = [f"http://localhost:{9000 + i}" for i in range(min(num_gpus, 4))]
            else:
                endpoints = ["http://localhost:9000"]
        
        logger.info(f"Using inference endpoints: {endpoints}")
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