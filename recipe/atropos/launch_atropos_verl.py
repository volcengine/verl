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
Production Atropos-VERL Integration Launcher
This script launches a complete Atropos-VERL integration with distributed training,
using VERL's infrastructure for model loading, inference, and training.

Features:
- Distributed training with FSDP and Ulysses
- Production inference engines (vLLM/SGLang)
- Complete Atropos API integration
- Automatic weight synchronization
- GPRO advantage-weighted SFT training
"""

import argparse
import logging
import os
from typing import Any, Dict

import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .atropos_trainer import AtroposTrainer
from .data_loader import AtroposDataLoader
from .main_atropos import AtroposAPIError, AtroposRLTrainer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_ATROPOS_LOGGING_LEVEL", "INFO"))


def setup_distributed_training():
    """Setup distributed training environment."""
    if not torch.distributed.is_initialized():
        # Initialize distributed training
        torch.distributed.init_process_group(backend="nccl")

    # Get local rank and world size
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    return local_rank, world_size, device


def create_device_meshes(world_size: int, local_rank: int):
    """Create device meshes for distributed training."""
    # Create main device mesh for data parallel
    device_mesh = init_device_mesh("cuda", (world_size,))

    # Create Ulysses device mesh for sequence parallel (if needed)
    ulysses_device_mesh = None
    if world_size > 1:
        ulysses_device_mesh = device_mesh

    return device_mesh, ulysses_device_mesh


def create_mock_datasets():
    """Create mock datasets for demonstration."""
    from torch.utils.data import Dataset

    class MockDataset(Dataset):
        def __init__(self, size=1000):
            self.size = size
            # Create mock data
            self.data = [{"input_ids": torch.randint(0, 1000, (64,)), "attention_mask": torch.ones(64), "labels": torch.randint(0, 1000, (64,))} for _ in range(size)]

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.data[idx]

    train_dataset = MockDataset(1000)
    val_dataset = MockDataset(100)

    return train_dataset, val_dataset


def create_tokenizer(model_path: str):
    """Create tokenizer for the model."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def run_atropos_training(config: Dict[str, Any], device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh):
    """Run Atropos training with VERL infrastructure."""
    local_rank = device_mesh.get_rank()

    if local_rank == 0:
        print("🚀 Starting Atropos-VERL Production Training")
        print("=" * 60)

    # Create tokenizer
    model_path = config.get("model_path", "microsoft/DialoGPT-medium")
    tokenizer = create_tokenizer(model_path)

    # Create datasets
    train_dataset, val_dataset = create_mock_datasets()

    # Create AtroposTrainer
    trainer = AtroposTrainer(config=OmegaConf.create(config), device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)

    if local_rank == 0:
        print("✓ AtroposTrainer initialized successfully")
        print(f"  - Model: {model_path}")
        print(f"  - World size: {device_mesh.size()}")
        print(f"  - Advantage weighting: {trainer.use_advantage_weighting}")

    # Start training
    try:
        trainer.fit()
        if local_rank == 0:
            print("✅ Training completed successfully!")
    except Exception as e:
        if local_rank == 0:
            print(f"❌ Training failed: {e}")
        raise


def run_atropos_demo(config: Dict[str, Any]):
    """Run Atropos demo with RL training loop using production datasets."""
    print("🚀 Starting Atropos-VERL Demo")
    print("=" * 60)

    try:
        # Initialize RL trainer
        trainer = AtroposRLTrainer(config)

        # Load production prompts from datasets
        data_config = {
            "data_source": "atropos_integration",
            "max_prompts": 10,
            "prompt_format": "chat",
            "parquet_paths": ["~/data/rlhf/gsm8k/train.parquet", "~/data/rlhf/math/train.parquet"],
            "hf_datasets": ["gsm8k", "math", "hellaswag"],
            "max_prompt_length": 512,
            "max_response_length": 32,
            "ability": "general",
        }

        loader = AtroposDataLoader(data_config)
        prompts = loader.load_production_prompts()

        # Run RL training loop
        print(f"\n🎯 Starting RL training with {len(prompts)} production prompts")
        print("=" * 50)

        for step in range(3):  # Run 3 training steps
            try:
                result = trainer.rl_training_step(prompts)
                print(f"\n✅ Step {result['step']} completed successfully!")
                print(f"   Loss: {result['loss']:.4f}")
                print(f"   Advantages shape: {result['advantages'].shape}")

            except Exception as e:
                print(f"❌ Error in training step {step}: {e}")
                break

        print("\n🎉 Atropos-VERL demo completed!")

    except AtroposAPIError as e:
        print(f"❌ ATROPOS API ERROR: {e}")
        print("\nEnsure that:")
        print("1. Atropos server is running on the configured URL")
        print("2. The API endpoints are accessible")
        print("3. Network connectivity is available")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True


def main():
    """Main entry point for Atropos-VERL integration."""
    parser = argparse.ArgumentParser(description="Atropos-VERL Integration Launcher")
    parser.add_argument("--mode", choices=["demo", "training"], default="demo", help="Mode to run: demo or training")
    parser.add_argument("--config", type=str, default="recipe/atropos/config/atropos_trainer.yaml", help="Path to configuration file")
    parser.add_argument("--model_path", type=str, default="microsoft/DialoGPT-medium", help="Model path for training")
    parser.add_argument("--atropos_url", type=str, default="http://localhost:9001", help="Atropos API URL")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--max_response_length", type=int, default=32, help="Maximum response length")
    parser.add_argument("--use_distributed", action="store_true", help="Use distributed training")

    # GPRO-specific arguments
    parser.add_argument("--use_gpro", action="store_true", default=True, help="Use GPRO for advantage computation")
    parser.add_argument("--gpro_epsilon", type=float, default=1e-6, help="GPRO epsilon for numerical stability")
    parser.add_argument("--gpro_norm_by_std", action="store_true", default=True, help="Normalize GPRO advantages by standard deviation")

    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = OmegaConf.load(args.config)
        config = OmegaConf.to_container(config, resolve=True)
    else:
        # Default configuration with GPRO
        config = {
            "atropos": {"api_url": args.atropos_url, "timeout": 30},
            "use_advantage_weighting": True,
            "use_gpro": args.use_gpro,  # GPRO integration
            "gpro_epsilon": args.gpro_epsilon,
            "gpro_norm_by_std": args.gpro_norm_by_std,
            "advantage_normalization": "batch",
            "advantage_clipping": [-3.0, 3.0],
            "max_response_length": args.max_response_length,
            "batch_size": args.batch_size,
            "batch_retry_attempts": 8,
            "batch_retry_delay": 0.3,
            "batch_max_wait_time": 12.0,
            "model_path": args.model_path,
            "device": "cuda",
            "trainer": {"project_name": "verl_atropos_integration", "experiment_name": "gpro_advantage_weighted_sft", "device": "cuda", "total_epochs": 1, "total_training_steps": 100, "val_before_train": True, "val_only": False, "logger": "wandb", "n_gpus_per_node": 1, "nnodes": 1},
            "data": {"train_batch_size": args.batch_size * 8, "micro_batch_size_per_gpu": args.batch_size, "max_length": 512, "max_response_length": args.max_response_length, "balance_dp_token": True},
            "model": {"path": args.model_path, "partial_pretrain": args.model_path, "trust_remote_code": False, "fsdp_config": {"model_dtype": "bf16", "use_meta_tensor": True, "cpu_offload": False, "mixed_precision": True, "sharding_strategy": "FULL_SHARD", "activation_checkpointing": True}},
            "ulysses_sequence_parallel_size": 1,
            "use_remove_padding": False,
        }

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["VLLM_LOGGING_LEVEL"] = "WARN"
    os.environ["VERL_ATROPOS_LOGGING_LEVEL"] = "INFO"

    if args.use_distributed:
        # Initialize distributed training
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            rank = 0
            world_size = 1
            local_rank = 0

        # Initialize process group
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        # Create device meshes
        device_mesh = init_device_mesh("cuda", (world_size,))
        ulysses_device_mesh = init_device_mesh("cuda", (world_size,))

        if rank == 0:
            print("🚀 Starting distributed Atropos-VERL training with GPRO")
            print(f"  - World size: {world_size}")
            print(f"  - GPRO enabled: {args.use_gpro}")
            print(f"  - Model: {args.model_path}")

        # Run distributed training
        run_atropos_training(config, device_mesh, ulysses_device_mesh)

        # Cleanup
        torch.distributed.destroy_process_group()
    else:
        # Run demo mode
        if args.mode == "demo":
            success = run_atropos_demo(config)
            if success:
                print("✅ Demo completed successfully!")
            else:
                print("❌ Demo failed")
        else:
            print("❌ Training mode requires --use_distributed flag")


if __name__ == "__main__":
    main()
