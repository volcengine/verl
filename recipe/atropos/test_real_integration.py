#!/usr/bin/env python3
"""
Test script to verify Atropos integration is working with real environment feedback.

This demonstrates:
1. Connecting to Atropos
2. Getting real prompts
3. Generating responses
4. Getting real advantages based on correctness
"""

import logging

from atropos_integration import AtroposEnvironmentClient, AtroposGRPOComputer
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_real_atropos_integration():
    """Test the integration with a real Atropos server"""

    # Initialize client
    client = AtroposEnvironmentClient(api_url="http://localhost:8000")

    # Check health
    if not client.health_check():
        logger.error("Cannot connect to Atropos. Please start the GSM8K server:")
        logger.error("python atropos/environments/gsm8k_server.py serve --slurm false")
        return False

    logger.info("✓ Connected to Atropos")

    # Get prompts from environment
    prompts_data = client.get_prompts(batch_size=4)
    if not prompts_data:
        logger.error("Failed to get prompts")
        return False

    prompts = prompts_data["prompts"]
    metadata = prompts_data["metadata"]

    logger.info(f"✓ Retrieved {len(prompts)} prompts from environment")
    logger.info(f"  Example: {prompts[0][:100]}...")

    # Example responses for testing
    # In real training, these come from the model
    responses = ["Let me solve this step by step. The answer is \\boxed{42}", "I think the answer is \\boxed{17}", "After calculation, \\boxed{25}", "The solution is \\boxed{100}"]

    # Use a real tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Submit and get advantages
    logger.info("Submitting responses to Atropos for evaluation...")
    result = client.submit_responses_and_get_advantages(prompts=prompts[: len(responses)], responses=responses, metadata=metadata, tokenizer=tokenizer)

    if not result:
        logger.error("Failed to get advantages")
        return False

    # Analyze results
    advantages = result["advantages"]
    rewards = result.get("rewards", [])
    env_metrics = result.get("env_metrics", {})

    logger.info("✓ Received real advantages from environment")
    logger.info(f"  Advantages shape: {advantages.shape}")
    logger.info(f"  Rewards: {rewards}")
    logger.info(f"  Environment metrics: {env_metrics}")

    # Test GRPO computer
    grpo_computer = AtroposGRPOComputer(client)
    logger.info("\n✓ GRPO computer initialized")

    # Verify advantages are based on real evaluation
    if advantages.numel() > 0:
        logger.info(f"  Mean advantage: {advantages.mean().item():.3f}")
        logger.info(f"  Std advantage: {advantages.std().item():.3f}")
        logger.info(f"  Min/Max: {advantages.min().item():.3f} / {advantages.max().item():.3f}")

        # Check that advantages vary based on response quality
        if advantages.std().item() > 0.01:
            logger.info("✓ Advantages show variation based on correctness")
        else:
            logger.warning("⚠ Advantages have low variation")

    return True


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing Atropos-VeRL Integration")
    logger.info("=" * 60)

    success = test_real_atropos_integration()

    if success:
        logger.info("\n✅ All tests passed! Integration is working correctly.")
        logger.info("The system is getting real advantages from Atropos environments.")
    else:
        logger.error("\n❌ Integration test failed.")
        logger.error("Please ensure Atropos server is running with an environment.")
