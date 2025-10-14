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
Example training script for GRPO with Atropos GSM8K environment.

This script trains a language model with GRPO using feedback from the Atropos GSM8K environment.
"""

import logging
import sys
from transformers import AutoTokenizer

from verl.trainer.main_ppo import get_args_parser, process_args
from verl.utils.import_utils import import_external_libs

from recipe.atropos.grpo_atropos_trainer import RayGRPOAtroposTrainer

logger = logging.getLogger(__name__)


def verify_atropos_connection(api_url: str = "http://localhost:9001") -> bool:
    """Verify that Atropos API is accessible."""
    import requests
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            logger.info(f"Successfully connected to Atropos at {api_url}")
            return True
        else:
            logger.error(f"Atropos health check failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Cannot connect to Atropos at {api_url}: {e}")
        return False


def main():
    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Process arguments
    config = process_args(args, seed=42)
    
    # Override trainer class to use Atropos GRPO trainer
    config.trainer_cls = RayGRPOAtroposTrainer
    
    # Ensure we're using GRPO with Atropos
    config.algorithm.adv_estimator = "grpo_atropos"
    config.algorithm.use_critic = False
    
    # Verify Atropos connection
    api_url = config.trainer.get("atropos", {}).get("api_url", "http://localhost:9001")
    if not verify_atropos_connection(api_url):
        logger.error("Failed to connect to Atropos. Please ensure Atropos server is running.")
        logger.error("Start it with: python environments/gsm8k_server.py serve --slurm false")
        sys.exit(1)
    
    # Import external libraries
    import_external_libs()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize trainer
    trainer = config.trainer_cls(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=config.role_worker_mapping,
        resource_pool_manager=config.resource_pool_manager
    )
    
    # Log training configuration
    logger.info("Starting GRPO training with Atropos integration")
    logger.info(f"Model: {config.model.path}")
    logger.info(f"Atropos API: {api_url}")
    logger.info(f"Advantage estimator: {config.algorithm.adv_estimator}")
    logger.info(f"Number of epochs: {config.trainer.total_epochs}")
    
    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    # Log final metrics
    if hasattr(trainer, "final_metrics"):
        logger.info("Final training metrics:")
        for key, value in trainer.final_metrics.items():
            logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    main()
