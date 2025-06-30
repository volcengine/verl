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
Test script for Atropos-VeRL integration.

This script validates the integration between VeRL and Atropos.
"""

import sys
from pathlib import Path
import torch
import logging
from transformers import AutoTokenizer

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from recipe.atropos.atropos_integration import AtroposConfig, AtroposEnvironmentClient, AtroposGRPOComputer
from verl.trainer.ppo.core_algos import compute_grpo_atropos_advantage, AdvantageEstimator

logger = logging.getLogger(__name__)


def test_atropos_client():
    """Test Atropos client connectivity and basic operations."""
    print("Testing Atropos client...")
    
    config = AtroposConfig(
        api_url="http://localhost:9001",
        timeout=10
    )
    
    try:
        client = AtroposEnvironmentClient(config)
        print("✓ Client initialized successfully")
        
        # Test with sample data
        prompts = ["What is 2+2?", "What is 5*3?"]
        responses = ["4", "15"]
        
        advantages, metrics = client.submit_responses_and_get_advantages(
            prompts, responses
        )
        
        if advantages is not None:
            print("✓ Successfully received advantages from Atropos")
            print(f"  Metrics: {metrics}")
        else:
            print("✗ No advantages received (API might be returning None)")
            
    except Exception as e:
        print(f"✗ Client test failed: {e}")
        return False
    
    return True


def test_grpo_computer():
    """Test GRPO computer with Atropos integration."""
    print("\nTesting GRPO computer...")
    
    config = AtroposConfig(
        api_url="http://localhost:9001",
        use_advantages=True,
        fallback_to_standard=True
    )
    
    try:
        computer = AtroposGRPOComputer(config)
        print("✓ GRPO computer initialized")
        
        # Create sample tensors
        batch_size = 2
        seq_length = 10
        
        prompts = torch.randint(0, 1000, (batch_size, seq_length))
        responses = torch.randint(0, 1000, (batch_size, seq_length))
        scores = torch.randn(batch_size)
        response_mask = torch.ones(batch_size, seq_length)
        
        # Mock tokenizer
        class MockTokenizer:
            def decode(self, tokens, skip_special_tokens=True):
                return " ".join(str(t.item()) if hasattr(t, 'item') else str(t) for t in tokens)
        
        tokenizer = MockTokenizer()
        
        # Compute advantages
        advantages, metrics = computer.compute_advantages_with_overrides(
            prompts=prompts,
            responses=responses,
            scores=scores,
            tokenizer=tokenizer,
            response_mask=response_mask
        )
        
        print("✓ Advantages computed successfully")
        print(f"  Shape: {advantages.shape}")
        print(f"  Metrics: {metrics}")
        
    except Exception as e:
        print(f"✗ GRPO computer test failed: {e}")
        return False
    
    return True


def test_advantage_estimator():
    """Test the registered grpo_atropos advantage estimator."""
    print("\nTesting grpo_atropos advantage estimator...")
    
    try:
        # Check if estimator is registered
        from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY
        
        if "grpo_atropos" in ADV_ESTIMATOR_REGISTRY:
            print("✓ grpo_atropos estimator is registered")
        else:
            print("✗ grpo_atropos estimator not found in registry")
            return False
        
        # Test the estimator
        batch_size = 4
        response_length = 20
        
        token_level_rewards = torch.randn(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = torch.arange(batch_size).numpy()
        
        advantages, returns = compute_grpo_atropos_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=True
        )
        
        print("✓ Advantage computation successful")
        print(f"  Advantages shape: {advantages.shape}")
        print(f"  Returns shape: {returns.shape}")
        
        # Test with token-level advantages override
        token_level_advantages = torch.randn(batch_size, response_length)
        advantages_override, returns_override = compute_grpo_atropos_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            token_level_advantages=token_level_advantages
        )
        
        print("✓ Advantage override successful")
        
    except Exception as e:
        print(f"✗ Advantage estimator test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Atropos-VeRL Integration Tests")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    tests = [
        ("Atropos Client", test_atropos_client),
        ("GRPO Computer", test_grpo_computer),
        ("Advantage Estimator", test_advantage_estimator)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    # Overall result
    all_passed = all(success for _, success in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. ✗")
        print("\nNote: If Atropos server is not running, start it with:")
        print("  python environments/gsm8k_server.py serve --slurm false")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())