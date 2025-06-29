#!/usr/bin/env python3
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
Main script for Atropos-VeRL integration with GSM8K environment

This script demonstrates:
- Training a model using Atropos GSM8K environment
- Real environment feedback for mathematical reasoning
- Token-level advantage computation
- Improved metrics tracking
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import ray
import torch
from omegaconf import DictConfig, OmegaConf

# Add VeRL to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atropos_ray_trainer import RayAtroposTrainer

from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayRoleWorkerGroup
from verl.utils.tracking import init_tracking
from verl.workers.actor.fsdp import FSDPPPOActor
from verl.workers.critic.fsdp import FSDPCritic
from verl.workers.reward_model.fsdp import FSDPRewardModel
from verl.workers.rollout.fsdp_workers import ActorRolloutRefWorker
from verl.workers.rollout.vllm_rollout import vLLMRollout

logger = logging.getLogger(__name__)


def create_default_config():
    """Create default configuration for Atropos GSM8K training"""
    config = {
        "trainer": {
            "project_name": "verl_atropos_gsm8k",
            "experiment_name": "atropos_gsm8k_integration",
            "logger": "wandb",  # or "tensorboard"
            "default_local_dir": "/tmp/verl_atropos_checkpoints",
            "total_epochs": 3,
            "total_training_steps": 1000,
            "save_freq": 100,
            "eval_freq": 50,
            "profile_steps": [],
            "balance_batch": True,
            # Atropos configuration
            "atropos": {
                "api_url": "http://localhost:9001",
                "timeout": 30,
                "use_advantages": True,
                "fallback_to_grpo": True,
                "retry_attempts": 10,
                "retry_delay": 0.5,
                "max_wait_time": 30.0,
            },
        },
        "algorithm": {
            # PPO hyperparameters
            "gamma": 1.0,
            "lam": 0.95,
            "adv_estimator": "grpo_atropos",  # Use our custom estimator
            "kl_penalty": "kl",
            "kl_coeff": 0.05,
            "clip_range": 0.2,
            "value_loss_coeff": 1.0,
            "entropy_coeff": 0.01,
            "max_grad_norm": 1.0,
            "enable_ptx_loss": False,
            # GRPO specific
            "use_kl_in_reward": True,
            "use_reference_policy": True,
            "use_critic": False,  # GRPO doesn't use critic
        },
        "model": {
            # Model selection - can be overridden
            "partial_pretrain": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "model_name": "qwen2.5_math_1.5b",
        },
        "actor_rollout_ref": {
            # Actor configuration
            "actor": {
                "strategy": "fsdp",
                "ppo_micro_batch_size": 2,
                "loss_agg_mode": "sum",
                "fsdp": {
                    "param_offload": False,
                    "grad_offload": False,
                    "optimizer_offload": False,
                },
                "optim": {
                    "type": "AdamW",
                    "lr": 1e-5,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                },
            },
            # Rollout configuration (using vLLM)
            "rollout": {
                "name": "vllm",
                "n": 1,  # Number of response per prompt
                "batch_size": 16,
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.5,
                "use_async": False,
            },
            # Reference model configuration
            "ref": {
                "log_prob_micro_batch_size": 8,
                "fsdp": {
                    "param_offload": False,
                },
            },
        },
        "critic": {
            # Critic configuration (disabled for GRPO)
            "enable": False,
        },
        "reward_model": {
            # We use Atropos environments for rewards
            "enable": False,
            "launch_reward_fn_async": False,
        },
        "data": {
            # Data configuration
            "train_files": ["gsm8k_train_messages.parquet"],
            "val_files": ["gsm8k_val_messages.parquet"],
            "train_batch_size": 128,
            "val_batch_size": 64,
            "max_prompt_length": 256,
            "num_workers": 4,
            "pin_memory": True,
        },
        # Distributed training configuration
        "distributed": {
            "backend": "nccl",
            "timeout_s": 1800,
        },
        # Resource allocation
        "n_gpus_per_node": torch.cuda.device_count() if torch.cuda.is_available() else 1,
    }

    return OmegaConf.create(config)


def main(config: DictConfig):
    """Main training loop for Atropos-GSM8K integration"""

    # Initialize Ray with GPU support
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                }
            }
        )

    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available! Training will be slow on CPU.")

    # Initialize tracking
    if config.trainer.logger:
        init_tracking(project_name=config.trainer.project_name, experiment_name=config.trainer.experiment_name, backend=config.trainer.logger)

    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))

    # Prepare data if needed
    train_data_path = prepare_gsm8k_data(config)
    config.data.train_files = [train_data_path]

    # Get number of GPUs per node
    n_gpus_per_node = config.get("n_gpus_per_node", torch.cuda.device_count())
    if n_gpus_per_node == 0:
        n_gpus_per_node = 1
        logger.warning("No GPUs detected, using CPU (not recommended)")

    # Create resource pool with GPU support
    resource_pool = RayResourcePool(process_on_nodes=[{"hostname": "localhost", "world_size": n_gpus_per_node}], use_gpu=torch.cuda.is_available(), name_prefix="atropos")

    # Create worker classes with proper initialization
    actor_rollout_cls = RayClassWithInitArgs(
        cls=ActorRolloutRefWorker,
        args=(FSDPPPOActor, vLLMRollout, None),  # No ref policy in same worker
    )

    ref_policy_cls = RayClassWithInitArgs(
        cls=ActorRolloutRefWorker,
        args=(None, None, FSDPPPOActor),  # Only ref policy
    )

    critic_cls = RayClassWithInitArgs(cls=FSDPCritic) if config.algorithm.get("use_critic", False) else None

    rm_cls = RayClassWithInitArgs(cls=FSDPRewardModel) if config.reward_model.get("enable", False) else None

    # Define worker mapping
    role_worker_mapping = {
        "actor_rollout": RayRoleWorkerGroup(
            name="actor_rollout",
            resource_pool=resource_pool,
            ray_cls=actor_rollout_cls,
        ),
        "ref": RayRoleWorkerGroup(
            name="ref",
            resource_pool=resource_pool,
            ray_cls=ref_policy_cls,
        )
        if config.algorithm.get("use_reference_policy", True)
        else None,
        "critic": RayRoleWorkerGroup(
            name="critic",
            resource_pool=resource_pool,
            ray_cls=critic_cls,
        )
        if config.algorithm.get("use_critic", False)
        else None,
        "rm": RayRoleWorkerGroup(
            name="rm",
            resource_pool=resource_pool,
            ray_cls=rm_cls,
        )
        if config.reward_model.get("enable", False)
        else None,
    }

    # Remove None values
    role_worker_mapping = {k: v for k, v in role_worker_mapping.items() if v is not None}

    # Create trainer
    trainer = RayAtroposTrainer(
        config=config,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool,
    )

    # Test Atropos connectivity before training
    logger.info("Testing Atropos API connectivity...")
    if not trainer.atropos_client.test_connectivity():
        logger.error("Cannot connect to Atropos API. Please ensure:\n1. Atropos server is running (atroposlib process)\n2. GSM8K environment is registered\n3. API is accessible at the configured URL\n\nTo start Atropos:\ncd /path/to/atropos && python -m atroposlib.api")
        return

    logger.info("Successfully connected to Atropos API!")

    # Start training
    logger.info("Starting Atropos-GSM8K training...")
    trainer.fit()

    logger.info("Training completed!")

    # Shutdown
    ray.shutdown()


def prepare_gsm8k_data(config: DictConfig) -> str:
    """Prepare GSM8K data in the format expected by VeRL"""
    import pandas as pd
    from datasets import load_dataset

    output_path = "/tmp/gsm8k_train_messages.parquet"

    # Check if data already exists
    if os.path.exists(output_path):
        logger.info(f"Using existing GSM8K data at {output_path}")
        return output_path

    logger.info("Preparing GSM8K training data...")

    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main", split="train")

    # Convert to message format expected by VeRL
    data_list = []
    for item in dataset:
        # Create prompt
        prompt = f"Question: {item['question']}\n\nPlease solve this step-by-step:"

        # Format for VeRL's data loader
        data_list.append(
            {
                "prompt": prompt,
                "question": item["question"],
                "answer": item["answer"],  # Contains both reasoning and final answer
                "input_ids": None,  # Will be tokenized by dataloader
                "attention_mask": None,
                "position_ids": None,
            }
        )

    # Convert to DataFrame and save
    df = pd.DataFrame(data_list)
    df.to_parquet(output_path, index=False)

    logger.info(f"Saved {len(df)} GSM8K examples to {output_path}")

    # Also prepare validation data
    val_dataset = load_dataset("gsm8k", "main", split="test[:100]")  # Use first 100 for validation
    val_data_list = []

    for item in val_dataset:
        prompt = f"Question: {item['question']}\n\nPlease solve this step-by-step:"

        val_data_list.append(
            {
                "prompt": prompt,
                "question": item["question"],
                "answer": item["answer"],
            }
        )

    val_df = pd.DataFrame(val_data_list)
    val_output_path = "/tmp/gsm8k_val_messages.parquet"
    val_df.to_parquet(val_output_path, index=False)

    logger.info(f"Saved {len(val_df)} GSM8K validation examples to {val_output_path}")

    return output_path


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train model with Atropos GSM8K environment")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--atropos-url", type=str, default="http://localhost:9001", help="Atropos API URL")
    parser.add_argument("--model", type=str, help="Model name to train")
    parser.add_argument("--n-gpus", type=int, help="Number of GPUs to use")
    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        config = create_default_config()

    # Override config with command line arguments
    if args.atropos_url:
        config.trainer.atropos.api_url = args.atropos_url
    if args.model:
        config.model.partial_pretrain = args.model
        config.actor_rollout_ref.actor.model.path = args.model
        config.actor_rollout_ref.ref.model.path = args.model
    if args.n_gpus:
        config.n_gpus_per_node = args.n_gpus

    # Run main
    main(config)
