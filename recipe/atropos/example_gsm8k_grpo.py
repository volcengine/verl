#!/usr/bin/env python3
"""
Example: GRPO Training with Atropos GSM8K Environment

Demonstrates training a model on GSM8K math problems with real environment feedback.
Shows measurable improvement in problem-solving accuracy through GRPO with 
token-level advantages from Atropos.

Usage:
    # First start Atropos GSM8K server:
    python atropos/environments/gsm8k_server.py serve --slurm false
    
    # Then run this script:
    python recipe/atropos/example_gsm8k_grpo.py
"""

import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add VeRL to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from verl.utils.fs import makedirs
from verl.utils.model import update_model_cls
from verl.trainer.main_generation import datetime, timedelta
from .grpo_atropos_trainer import RayGRPOAtroposTrainer

logger = logging.getLogger(__name__)


def verify_atropos_connection(api_url: str = "http://localhost:8000") -> bool:
    """Verify Atropos server is running and healthy"""
    import requests
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            status = requests.get(f"{api_url}/status", timeout=5).json()
            logger.info(f"Atropos server healthy. Environment: {status.get('environment', 'unknown')}")
            return True
    except:
        pass
    return False


@hydra.main(config_path="config", config_name="gsm8k_grpo_example", version_base=None)
def main(config: DictConfig):
    """
    Main training loop for GRPO with Atropos GSM8K.
    
    Trains a model to solve math problems using real correctness feedback
    from the Atropos GSM8K environment.
    """
    # Verify Atropos is running
    atropos_url = config.trainer.atropos.api_url
    if not verify_atropos_connection(atropos_url):
        logger.error(
            f"Cannot connect to Atropos at {atropos_url}. "
            "Please start the GSM8K server first:\n"
            "python atropos/environments/gsm8k_server.py serve --slurm false"
        )
        sys.exit(1)
        
    # Set up directories
    output_dir = config.trainer.default_local_dir
    makedirs(output_dir)
    
    # Configure model
    model_config = config.actor_rollout_ref.model
    update_model_cls(model_config)
    
    # Log configuration
    logger.info("="*50)
    logger.info("GRPO-Atropos GSM8K Training")
    logger.info("="*50)
    logger.info(f"Model: {model_config.path}")
    logger.info(f"Atropos URL: {atropos_url}")
    logger.info(f"Group size: {config.algorithm.group_size}")
    logger.info(f"Batch size: {config.actor_rollout_ref.rollout.batch_size}")
    logger.info(f"Learning rate: {config.actor_rollout_ref.actor.optim.lr}")
    logger.info("="*50)
    
    # Track initial and best metrics
    initial_accuracy = None
    best_accuracy = 0.0
    
    # Initialize trainer
    from verl.trainer.main_generation import init_ray, make_replay_buffer
    from verl.single_controller.ray import RayResourcePool
    from verl.workers import make_rollout_workerg, make_actor_workerg
    
    # Setup Ray
    ray_config = {"runtime_env": {"env_vars": {"TOKENIZERS_PARALLELISM": "false"}}}
    init_ray(config.ray_init, ray_config)
    
    # Create resource pool
    resource_pool_manager = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes
    )
    
    # Create workers
    role_worker_mapping = {}
    
    # Actor workers
    role_worker_mapping["actor"] = make_actor_workerg(
        config=config.actor_rollout_ref.actor,
        resource_pool_manager=resource_pool_manager,
        model_config=config.actor_rollout_ref.model
    )
    
    # Rollout workers 
    role_worker_mapping["rollout"] = make_rollout_workerg(
        config=config.actor_rollout_ref.rollout,
        resource_pool_manager=resource_pool_manager,
        model_config=config.actor_rollout_ref.model,
        actor_worker=role_worker_mapping["actor"]
    )
    
    # Reference policy workers if needed
    if config.actor_rollout_ref.ref is not None:
        from verl.workers.ref_worker import ReferenceWorker
        role_worker_mapping["ref"] = ReferenceWorker(
            config=config.actor_rollout_ref.ref,
            resource_pool_manager=resource_pool_manager,
            model_config=config.actor_rollout_ref.model
        )
        
    # Create trainer
    trainer = RayGRPOAtroposTrainer(
        config=config,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager
    )
    
    # Training loop with metric tracking
    logger.info("Starting GRPO training with Atropos GSM8K environment...")
    
    try:
        for epoch in range(config.trainer.total_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{config.trainer.total_epochs}")
            
            # Run training epoch
            epoch_metrics = trainer.train_epoch()
            
            # Track GSM8K accuracy
            if "atropos/gsm8k/correct_rate" in epoch_metrics:
                correct_rate = epoch_metrics["atropos/gsm8k/correct_rate"]
                
                if initial_accuracy is None:
                    initial_accuracy = correct_rate
                    logger.info(f"Initial accuracy: {correct_rate:.2%}")
                
                if correct_rate > best_accuracy:
                    best_accuracy = correct_rate
                    logger.info(f"New best accuracy: {correct_rate:.2%} (↑ {correct_rate - initial_accuracy:.2%})")
                else:
                    logger.info(f"Current accuracy: {correct_rate:.2%}")
                
            # Log advantage statistics
            if "grpo/advantages/mean" in epoch_metrics:
                adv_mean = epoch_metrics["grpo/advantages/mean"]
                adv_std = epoch_metrics.get("grpo/advantages/std", 0)
                logger.info(f"Advantages: mean={adv_mean:.3f}, std={adv_std:.3f}")
                
            # Save checkpoint
            if (epoch + 1) % config.trainer.save_freq == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
                trainer.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
            # Evaluation
            if (epoch + 1) % config.trainer.test_freq == 0:
                logger.info("Running evaluation...")
                eval_metrics = trainer.evaluate()
                
                if "eval/gsm8k/accuracy" in eval_metrics:
                    eval_acc = eval_metrics["eval/gsm8k/accuracy"]
                    logger.info(f"Evaluation accuracy: {eval_acc:.2%}")
                    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        trainer.cleanup()
        
        # Report final metrics
        logger.info("\n" + "="*50)
        logger.info("Training Summary")
        logger.info("="*50)
        if initial_accuracy is not None and best_accuracy > 0:
            improvement = best_accuracy - initial_accuracy
            logger.info(f"Initial accuracy: {initial_accuracy:.2%}")
            logger.info(f"Best accuracy: {best_accuracy:.2%}")
            logger.info(f"Total improvement: {improvement:.2%} ({improvement/initial_accuracy*100:.1f}% relative)")
        logger.info("Training completed successfully")


if __name__ == "__main__":
    main()