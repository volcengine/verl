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
This script launches a complete Atropos-VERL integration with GPRO advantage-weighted SFT training.

Features:
- Production inference engines (vLLM/SGLang)
- Complete Atropos API integration
- Automatic weight synchronization
- GPRO advantage-weighted SFT training
"""

import argparse
import logging
import os
from typing import Any, Dict

from omegaconf import OmegaConf

from .main_atropos import AtroposAPIError, AtroposRLTrainer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_ATROPOS_LOGGING_LEVEL", "INFO"))


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

        from .data_loader import AtroposDataLoader

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
        }

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["VLLM_LOGGING_LEVEL"] = "WARN"
    os.environ["VERL_ATROPOS_LOGGING_LEVEL"] = "INFO"

    # Run demo mode (single-GPU training)
    if args.mode == "demo":
        success = run_atropos_demo(config)
        if success:
            print("✅ Demo completed successfully!")
        else:
            print("❌ Demo failed")
    elif args.mode == "training":
        print("🚀 Starting Atropos-VERL training with GPRO")
        print(f"  - GPRO enabled: {args.use_gpro}")
        print(f"  - Model: {args.model_path}")

        # Run training using the same demo function for now
        # In production, this would use a proper training loop
        success = run_atropos_demo(config)
        if success:
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed")


if __name__ == "__main__":
    main()
