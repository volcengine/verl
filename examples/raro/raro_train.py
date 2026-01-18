#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Example script for training with RARO (Relativistic Adversarial Reasoning Optimization).

This script demonstrates how to use the RARO reward manager for adversarial training
of reasoning models following the "Escaping the Verifier" paper.

Usage:
    python raro_train.py --config path/to/raro_config.yaml
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.raro_utils import RARODualPassRollout
from verl.workers.reward_manager import RARORewardManager, RAROReplayBuffer
from verl.workers.reward_manager.raro_prompts import RAROPrompts


def create_raro_reward_manager(tokenizer, config):
    """Create a RARO reward manager from configuration.

    Args:
        tokenizer: Tokenizer for text encoding/decoding
        config: Configuration dictionary

    Returns:
        Initialized RARORewardManager
    """
    return RARORewardManager(
        tokenizer=tokenizer,
        num_examine=config.get("num_examine", 1),
        tau_pol=config.get("tau_pol", 0.6),
        tau_crit=config.get("tau_crit", 0.55),
        replay_buffer_size=config.get("replay_buffer_size", 10000),
        replay_buffer_ratio=config.get("replay_buffer_ratio", 0.5),
        max_response_length=config.get("max_response_length", 4096),
        shuffle_answers=config.get("shuffle_answers", True),
        buffer_type=config.get("buffer_type", "fifo"),
    )


def main():
    parser = argparse.ArgumentParser(description="Train a model with RARO")

    # Model and data arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--config", type=str, default=None, help="Path to RARO config YAML")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="outputs/raro", help="Output directory")
    parser.add_argument("-- epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs")

    # RARO-specific arguments
    parser.add_argument("--tau_pol", type=float, default=0.6, help="Policy tie reward")
    parser.add_argument("--tau_crit", type=float, default=0.55, help="Critic tie reward")
    parser.add_argument("--lambda_pol", type=float, default=1.0 / 9.0, help="Policy loss weight")
    parser.add_argument("--lambda_crit", type=float, default=8.0 / 9.0, help="Critic loss weight")
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--replay_buffer_ratio", type=float, default=0.5, help="Replay buffer ratio")
    parser.add_argument("--n_rollouts", type=int, default=16, help="Number of GRPO rollouts")

    args = parser.parse_args()

    # Load tokenizer
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Create RARO reward manager
    raro_config = {
        "num_examine": 1,
        "tau_pol": args.tau_pol,
        "tau_crit": args.tau_crit,
        "replay_buffer_size": args.replay_buffer_size,
        "replay_buffer_ratio": args.replay_buffer_ratio,
        "max_response_length": 4096,
        "shuffle_answers": True,
        "buffer_type": "fifo",
    }

    reward_manager = create_raro_reward_manager(tokenizer, raro_config)

    print("RARO Reward Manager initialized successfully!")
    print(f"  Replay buffer size: {args.replay_buffer_size}")
    print(f"  Tau (Policy): {args.tau_pol}")
    print(f"  Tau (Critic): {args.tau_crit}")
    print(f"  Lambda (Policy): {args.lambda_pol}")
    print(f"  Lambda (Critic): {args.lambda_crit}")

    # TODO: Initialize trainer with RARO configuration
    # This would typically use RayPPOTrainer with custom configuration
    print("\nTo continue training, integrate with RayPPOTrainer:")
    print("  1. Load the RARO configuration YAML")
    print("  2. Initialize the trainer with the config")
    print("  3. Use the RARODualPassRollout wrapper for dual-pass generation")

    print("\nExample usage:")
    print("  from verl.trainer.ppo.ray_trainer import RayPPOTrainer")
    print("  from omegaconf import OmegaConf")
    print("")
    print("  # Load config")
    print("  cfg = OmegaConf.load('verl/trainer/config/raro.yaml')")
    print("")
    print("  # Create trainer")
    print("  trainer = RayPPOTrainer(")
    print("      config=cfg.trainer,")
    print("      tokenizer=tokenizer,")
    print("      reward_manager=reward_manager,")
    print("      ...")
    print("  )")
    print("")
    print("  # Train")
    print("  trainer.fit()")


if __name__ == "__main__":
    main()
