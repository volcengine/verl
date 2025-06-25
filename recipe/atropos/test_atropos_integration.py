#!/usr/bin/env python3
"""
Test script for Atropos-VeRL integration

This script tests the basic functionality without running full training.
"""

import sys
import logging
from pathlib import Path

# Add VeRL to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atropos_api_client import AtroposAPIClient, AtroposConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_atropos_connectivity():
    """Test basic connectivity to Atropos API"""
    config = AtroposConfig(
        api_url="http://localhost:9001",
        batch_size=4,
        max_token_len=512
    )
    
    client = AtroposAPIClient(config)
    
    # Test connectivity
    logger.info("Testing Atropos API connectivity...")
    if not client.test_connectivity():
        logger.error("Failed to connect to Atropos API")
        logger.info("\nTo start Atropos:")
        logger.info("1. cd /path/to/atropos")
        logger.info("2. python -m atroposlib.api --port 9001")
        logger.info("3. In another terminal: python environments/gsm8k_environment.py")
        return False
        
    logger.info("✓ Connected to Atropos API")
    
    # Test registration
    logger.info("\nTesting trainer registration...")
    if not client.register_trainer(starting_step=0, num_steps=100):
        logger.error("Failed to register trainer")
        return False
        
    logger.info(f"✓ Registered trainer with UUID: {client.trainer_uuid}")
    
    # Get environment info
    logger.info("\nGetting environment information...")
    env_info = client.get_environment_info()
    logger.info(f"Available environments: {env_info}")
    
    # Test data submission
    logger.info("\nTesting data submission...")
    test_tokens = [[1, 2, 3, 4, 5, 6, 7, 8]]
    test_masks = [[0, 0, 0, 1, 1, 1, 1, 1]]
    test_scores = [0.75]
    
    success = client.submit_scored_data(
        tokens=test_tokens * 4,  # Replicate to meet batch size
        masks=test_masks * 4,
        scores=test_scores * 4
    )
    
    if not success:
        logger.error("Failed to submit data")
        return False
        
    logger.info("✓ Successfully submitted test data")
    
    # Test batch retrieval
    logger.info("\nTesting batch retrieval...")
    batch = client.retrieve_batch()
    
    if batch is not None:
        logger.info(f"✓ Retrieved batch with {len(batch)} items")
        logger.info(f"Batch item keys: {list(batch[0].keys()) if batch else 'N/A'}")
    else:
        logger.warning("No batch available (this is normal if no environments are processing)")
        
    logger.info("\n✅ All tests passed!")
    return True


def test_advantage_estimator():
    """Test the custom grpo_atropos advantage estimator"""
    import torch
    import numpy as np
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn
    
    logger.info("\nTesting grpo_atropos advantage estimator...")
    
    # Get the estimator function
    try:
        adv_fn = get_adv_estimator_fn("grpo_atropos")
        logger.info("✓ Successfully loaded grpo_atropos estimator")
    except ValueError as e:
        logger.error(f"Failed to load estimator: {e}")
        return False
        
    # Test with sample data
    batch_size = 4
    seq_len = 10
    
    token_level_rewards = torch.randn(batch_size, seq_len)
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, 5:] = 1  # Last 5 tokens are response
    
    index = np.array([0, 0, 1, 1])  # Two groups
    
    # Compute advantages
    advantages, _ = adv_fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config={"norm_adv_by_std_in_grpo": True}
    )
    
    logger.info(f"✓ Computed advantages shape: {advantages.shape}")
    logger.info(f"✓ Advantages range: [{advantages.min():.3f}, {advantages.max():.3f}]")
    
    return True


if __name__ == "__main__":
    logger.info("Atropos-VeRL Integration Test")
    logger.info("=" * 40)
    
    # Test connectivity
    connectivity_ok = test_atropos_connectivity()
    
    # Test advantage estimator
    estimator_ok = test_advantage_estimator()
    
    if connectivity_ok and estimator_ok:
        logger.info("\n✅ All integration tests passed!")
    else:
        logger.info("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)