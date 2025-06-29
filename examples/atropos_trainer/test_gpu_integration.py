#!/usr/bin/env python3
"""
Test script for VeRL Atropos GPU Integration

This script validates that the Atropos integration works properly with GPU devices,
testing device placement, memory management, and tensor operations.
"""

import logging
import sys
from pathlib import Path

import torch

# Add VeRL to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from verl.utils.atropos_utils import AtroposDataConverter, get_device_for_tensor_ops

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gpu_device_detection():
    """Test GPU device detection and setup"""
    print("🔍 Testing GPU Device Detection...")

    # Test auto-detection
    device = get_device_for_tensor_ops()
    print(f"   Auto-detected device: {device}")

    # Test preferred device
    if torch.cuda.is_available():
        cuda_device = get_device_for_tensor_ops("cuda:0")
        print(f"   CUDA device: {cuda_device}")

        # Test invalid device handling
        invalid_device = get_device_for_tensor_ops("cuda:99")
        print(f"   Invalid device fallback: {invalid_device}")

    cpu_device = get_device_for_tensor_ops("cpu")
    print(f"   CPU device: {cpu_device}")

    print("✅ GPU device detection test passed\n")
    return device


def test_tensor_operations(device):
    """Test basic tensor operations on the target device"""
    print(f"🔧 Testing Tensor Operations on {device}...")

    # Create test tensors
    tokens = torch.randint(0, 1000, (4, 16), device=device)
    masks = torch.ones_like(tokens, dtype=torch.float, device=device)
    scores = torch.randn(4, device=device)

    print(f"   Created tensors on device: {tokens.device}")
    print(f"   Tokens shape: {tokens.shape}")
    print(f"   Masks shape: {masks.shape}")
    print(f"   Scores shape: {scores.shape}")

    # Test basic operations
    token_sum = tokens.sum()
    mask_mean = masks.mean()
    score_max = scores.max()

    print(f"   Operations completed: sum={token_sum}, mean={mask_mean}, max={score_max}")

    if device.type == "cuda":
        # Test GPU memory info
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        cached = torch.cuda.memory_reserved() / 1024**2  # MB
        print(f"   GPU Memory - Allocated: {allocated:.1f}MB, Cached: {cached:.1f}MB")

        # Clear cache
        torch.cuda.empty_cache()
        cached_after = torch.cuda.memory_reserved() / 1024**2
        print(f"   GPU Memory after cache clear - Cached: {cached_after:.1f}MB")

    print("✅ Tensor operations test passed\n")
    return tokens, masks, scores


def test_atropos_data_conversion(tokens, masks, scores, device):
    """Test Atropos data conversion with GPU tensors"""
    print("🔄 Testing Atropos Data Conversion...")

    # Test VeRL to Atropos conversion (should move to CPU)
    atropos_data = AtroposDataConverter.verl_to_atropos_batch(tokens=tokens, masks=masks, scores=scores)

    print(f"   Converted to Atropos format: {type(atropos_data)}")
    print(f"   Data keys: {list(atropos_data.keys())}")
    print(f"   Tokens type: {type(atropos_data['tokens'])}")

    # Create mock Atropos batch for reverse conversion
    mock_batch = {
        "batch": [
            {"tokens": atropos_data["tokens"][0], "masks": atropos_data["masks"][0], "advantages": [0.5] * len(atropos_data["tokens"][0]), "scores": [atropos_data["scores"][0]]},
            {"tokens": atropos_data["tokens"][1], "masks": atropos_data["masks"][1], "advantages": [0.3] * len(atropos_data["tokens"][1]), "scores": [atropos_data["scores"][1]]},
        ]
    }

    # Test Atropos to VeRL conversion (should move to target device)
    converted_tokens, converted_masks, converted_advantages, converted_scores = AtroposDataConverter.atropos_to_verl_batch(mock_batch, device=device)

    print("   Converted back to VeRL format")
    print(f"   Tokens device: {converted_tokens.device}, shape: {converted_tokens.shape}")
    print(f"   Masks device: {converted_masks.device}, shape: {converted_masks.shape}")
    print(f"   Advantages device: {converted_advantages.device}, shape: {converted_advantages.shape}")
    print(f"   Scores device: {converted_scores.device}, shape: {converted_scores.shape}")

    # Verify devices match
    assert converted_tokens.device == device, f"Tokens on wrong device: {converted_tokens.device} != {device}"
    assert converted_masks.device == device, f"Masks on wrong device: {converted_masks.device} != {device}"
    assert converted_advantages.device == device, f"Advantages on wrong device: {converted_advantages.device} != {device}"
    assert converted_scores.device == device, f"Scores on wrong device: {converted_scores.device} != {device}"

    print("✅ Atropos data conversion test passed\n")


def test_memory_cleanup(device):
    """Test GPU memory cleanup"""
    if device.type != "cuda":
        print("⏭️ Skipping memory cleanup test (not on CUDA)\n")
        return

    print("🧹 Testing GPU Memory Cleanup...")

    # Get initial memory state
    initial_allocated = torch.cuda.memory_allocated()
    initial_cached = torch.cuda.memory_reserved()

    # Create large tensors
    large_tensors = []
    for i in range(10):
        tensor = torch.randn(1000, 1000, device=device)
        large_tensors.append(tensor)

    after_alloc = torch.cuda.memory_allocated()
    after_cached = torch.cuda.memory_reserved()

    print(f"   Before: Allocated={initial_allocated / 1024**2:.1f}MB, Cached={initial_cached / 1024**2:.1f}MB")
    print(f"   After creating tensors: Allocated={after_alloc / 1024**2:.1f}MB, Cached={after_cached / 1024**2:.1f}MB")

    # Clear references
    del large_tensors

    # Force cleanup
    torch.cuda.empty_cache()

    final_allocated = torch.cuda.memory_allocated()
    final_cached = torch.cuda.memory_reserved()

    print(f"   After cleanup: Allocated={final_allocated / 1024**2:.1f}MB, Cached={final_cached / 1024**2:.1f}MB")

    print("✅ GPU memory cleanup test passed\n")


def main():
    """Run all GPU integration tests"""
    print("🚀 VeRL Atropos GPU Integration Test Suite")
    print("=" * 50)

    # System info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
    print()

    try:
        # Test 1: Device detection
        device = test_gpu_device_detection()

        # Test 2: Tensor operations
        tokens, masks, scores = test_tensor_operations(device)

        # Test 3: Data conversion
        test_atropos_data_conversion(tokens, masks, scores, device)

        # Test 4: Memory cleanup
        test_memory_cleanup(device)

        print("🎉 All GPU integration tests passed!")
        print("VeRL Atropos integration is ready for GPU training.")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
