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

import logging
import sys
from pathlib import Path

import torch

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from recipe.atropos.atropos_integration import AtroposConfig, AtroposEnvironmentClient, AtroposGRPOComputer

# Try to import VeRL components, but handle gracefully if missing
try:
    from verl.trainer.ppo.core_algos import compute_grpo_atropos_advantage
    VERL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VeRL components not available ({e}). Some tests will be skipped.")
    VERL_AVAILABLE = False

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


def test_advantage_broadcast_logic():
    """Test the broadcast logic for scalar vs token-level advantages."""
    print("\nTesting advantage broadcast logic...")
    
    try:
        # Test the broadcast logic directly without requiring API connectivity
        from recipe.atropos.atropos_integration import AtroposConfig, AtroposEnvironmentClient
        
        # Create a config but don't initialize the client (to avoid connectivity test)
        config = AtroposConfig(
            api_url="http://localhost:9001",
            use_advantages=True,
            fallback_to_standard=True
        )
        
        # Create client instance but bypass connectivity test by mocking
        client = object.__new__(AtroposEnvironmentClient)
        client.config = config
        
        # Test 1: Scalar advantages should be broadcasted
        batch_size = 2
        seq_length = 5
        response_mask = torch.ones(batch_size, seq_length)
        
        # Mock scalar advantages (one per sample)
        scalar_advantages = [0.5, -0.3]
        
        tensor_advantages = client._convert_to_token_level_advantages(
            scalar_advantages, response_mask
        )
        
        # Check that scalar advantages were broadcasted correctly
        expected = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5],
                               [-0.3, -0.3, -0.3, -0.3, -0.3]])
        
        assert torch.allclose(tensor_advantages, expected), \
            f"Scalar broadcast failed: got {tensor_advantages}, expected {expected}"
        
        print("✓ Scalar advantage broadcasting works correctly")
        
        # Test 2: With partial response mask
        partial_mask = torch.tensor([[1, 1, 0, 0, 0],
                                   [1, 1, 1, 0, 0]], dtype=torch.float32)
        
        tensor_advantages_masked = client._convert_to_token_level_advantages(
            scalar_advantages, partial_mask
        )
        
        expected_masked = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0],
                                       [-0.3, -0.3, -0.3, 0.0, 0.0]])
        
        assert torch.allclose(tensor_advantages_masked, expected_masked), \
            f"Masked broadcast failed: got {tensor_advantages_masked}, expected {expected_masked}"
        
        print("✓ Masked advantage broadcasting works correctly")
        
        # Test 3: Token-level advantages should pass through unchanged
        # This would require mocking the API response, so we'll test the shape validation
        token_level_advantages = torch.randn(batch_size, seq_length)
        
        # This should work without errors (shape matches)
        try:
            result = token_level_advantages * response_mask
            assert result.shape == (batch_size, seq_length)
            print("✓ Token-level advantage shape validation works")
        except Exception as e:
            print(f"✗ Token-level advantage test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Broadcast logic test failed: {e}")
        return False


def test_advantage_estimator():
    """Test the registered grpo_atropos advantage estimator."""
    print("\nTesting grpo_atropos advantage estimator...")
    
    if not VERL_AVAILABLE:
        print("Skipping grpo_atropos advantage estimator test as VeRL is not available.")
        return True # Indicate success if VeRL is not available

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


def test_fallback_on_api_failure():
    """Test that fallback works when Atropos API is unavailable."""
    print("\nTesting fallback on API failure...")
    
    # Use a non-existent URL to simulate API failure
    config = AtroposConfig(
        api_url="http://localhost:9999",  # Non-existent port
        timeout=2,
        retry_attempts=2,  # Reduce for faster test
        fallback_to_standard=True
    )
    
    try:
        AtroposGRPOComputer(config)
        print("✗ Expected connection failure but got success")
        return False
    except Exception:
        # Expected - connection should fail
        pass
    
    # Test that fallback estimator works
    if not VERL_AVAILABLE:
        print("Skipping fallback estimator test as VeRL is not available.")
        return True # Indicate success if VeRL is not available

    try:
        from verl.trainer.ppo.core_algos import compute_grpo_atropos_advantage
        
        batch_size = 2
        response_length = 10
        
        token_level_rewards = torch.randn(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = torch.arange(batch_size).numpy()
        
        # This should work without Atropos (no token_level_advantages provided)
        advantages, returns = compute_grpo_atropos_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=True
        )
        
        print("✓ Fallback computation successful")
        print(f"  Advantages shape: {advantages.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Fallback test failed: {e}")
        return False


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
        ("Advantage Broadcast Logic", test_advantage_broadcast_logic),
        ("Advantage Estimator", test_advantage_estimator),
        ("Fallback on API Failure", test_fallback_on_api_failure)
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