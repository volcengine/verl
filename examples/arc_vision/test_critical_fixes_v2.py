#!/usr/bin/env python3
"""Test the two critical fixes for training with better test cases."""

import sys
from pathlib import Path

# Add the examples directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_vision.arc_vision_custom_reward import arc_vision_compute_reward

def test_negative_rewards():
    """Test that negative rewards can propagate (no clamping to 0)."""
    print("Testing negative reward propagation...")
    
    # Case 1: Unnecessary tool use - high confidence but used tools
    # This should trigger the "unnecessary_tool" penalty
    result = arc_vision_compute_reward(
        data_source="arc_vision",
        solution_str="""<reasoning>
I can clearly see the element at coordinates [0.1, 0.1, 0.2, 0.2].
</reasoning>

<tool_call>
name: zoom
keypoint: [0.15, 0.15]
</tool_call>

<bbox>[0.1, 0.1, 0.2, 0.2]</bbox>""",
        ground_truth=[0.8, 0.8, 0.9, 0.9],  # Far from prediction
        extra_info={"index": 0}
    )
    
    reward = result["reward"]
    print(f"Case 1 - Unnecessary tool penalty:")
    print(f"  Reward: {reward:.6f}")
    print(f"  Components: r_task={result.get('r_task', 'N/A'):.3f}, r_tool={result.get('r_tool', 'N/A'):.3f}, r_gate={result.get('r_gate', 'N/A'):.3f}")
    print(f"  Gate penalties: {result.get('gate_penalties', [])}")
    
    # Case 2: Multiple penalties - unnecessary tools AND poor detection
    result2 = arc_vision_compute_reward(
        data_source="arc_vision",
        solution_str="""<reasoning>
I can see this element clearly.
</reasoning>

<tool_call>
name: zoom
keypoint: [0.5, 0.5]
</tool_call>

<tool_call>
name: zoom
keypoint: [0.5, 0.5]
</tool_call>

<tool_call>
name: zoom
keypoint: [0.5, 0.5]
</tool_call>

<tool_call>
name: wait
duration: 1
</tool_call>

<bbox>[0.1, 0.1, 0.2, 0.2]</bbox>""",
        ground_truth=[0.8, 0.8, 0.9, 0.9],
        extra_info={"index": 1},
        reward_weights={"task": 0.5, "tool": 0.2, "gate": 0.3}  # Increase gate weight
    )
    
    reward2 = result2["reward"]
    print(f"\nCase 2 - Multiple penalties with custom weights:")
    print(f"  Reward: {reward2:.6f}")
    print(f"  Components: r_task={result2.get('r_task', 'N/A'):.3f}, r_tool={result2.get('r_tool', 'N/A'):.3f}, r_gate={result2.get('r_gate', 'N/A'):.3f}")
    print(f"  Gate penalties: {result2.get('gate_penalties', [])}")
    
    if reward < 0 or reward2 < 0:
        print("\n✅ Negative rewards can propagate (no clamping)")
        return True
    else:
        print("\n❌ ERROR: Negative rewards are not going negative")
        return False

def test_excessive_tools_penalty():
    """Test that excessive tools penalty triggers at >2 tools."""
    print("\nTesting excessive tools penalty threshold...")
    
    # Use custom weights that make the penalty more impactful
    custom_weights = {"task": 0.5, "tool": 0.1, "gate": 0.4}
    
    # Case 1: Exactly 2 tools - should NOT trigger penalty
    response_2_tools = """<reasoning>
I need to zoom in twice to see the element clearly.
</reasoning>

<tool_call>
name: zoom
keypoint: [0.5, 0.5]
</tool_call>

<tool_call>
name: zoom
keypoint: [0.6, 0.6]
</tool_call>

<bbox>[0.5, 0.5, 0.6, 0.6]</bbox>"""
    
    result_2 = arc_vision_compute_reward(
        data_source="arc_vision",
        solution_str=response_2_tools,
        ground_truth=[0.5, 0.5, 0.6, 0.6],
        extra_info={"index": 1},
        reward_weights=custom_weights
    )
    
    # Case 2: 3 tools - SHOULD trigger penalty
    response_3_tools = """<reasoning>
I need multiple tools to find this element.
</reasoning>

<tool_call>
name: zoom
keypoint: [0.5, 0.5]
</tool_call>

<tool_call>
name: wait
duration: 1
</tool_call>

<tool_call>
name: inspect
region: [0.5, 0.5, 0.6, 0.6]
</tool_call>

<bbox>[0.5, 0.5, 0.6, 0.6]</bbox>"""
    
    result_3 = arc_vision_compute_reward(
        data_source="arc_vision",
        solution_str=response_3_tools,
        ground_truth=[0.5, 0.5, 0.6, 0.6],
        extra_info={"index": 2},
        reward_weights=custom_weights
    )
    
    # Compare rewards - 3 tools should have lower reward due to penalty
    reward_2_tools = result_2["reward"]
    reward_3_tools = result_3["reward"]
    
    print(f"With custom weights (gate=0.4):")
    print(f"  2 tools: {reward_2_tools:.3f} (r_gate={result_2.get('r_gate', 0):.3f})")
    print(f"  3 tools: {reward_3_tools:.3f} (r_gate={result_3.get('r_gate', 0):.3f})")
    
    # Also test with default weights for comparison
    result_2_default = arc_vision_compute_reward(
        data_source="arc_vision",
        solution_str=response_2_tools,
        ground_truth=[0.5, 0.5, 0.6, 0.6],
        extra_info={"index": 3}
    )
    
    result_3_default = arc_vision_compute_reward(
        data_source="arc_vision",
        solution_str=response_3_tools,
        ground_truth=[0.5, 0.5, 0.6, 0.6],
        extra_info={"index": 4}
    )
    
    print(f"\nWith default weights (gate=0.1):")
    print(f"  2 tools: {result_2_default['reward']:.3f} (r_gate={result_2_default.get('r_gate', 0):.3f})")
    print(f"  3 tools: {result_3_default['reward']:.3f} (r_gate={result_3_default.get('r_gate', 0):.3f})")
    
    if reward_3_tools < reward_2_tools:
        print("\n✅ Excessive tools penalty triggers correctly at >2 tools")
        return True
    else:
        print("\n❌ ERROR: Excessive tools penalty not effective enough")
        print("   The penalty is applied but tool reward increase outweighs it.")
        print("   Consider increasing gate weight or reducing tool weight in training.")
        return False

def main():
    print("=== Testing Critical Fixes V2 ===\n")
    
    tests_passed = 0
    
    # Test 1: Negative rewards
    if test_negative_rewards():
        tests_passed += 1
    
    # Test 2: Excessive tools penalty
    if test_excessive_tools_penalty():
        tests_passed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {tests_passed}/2")
    
    if tests_passed == 2:
        print("✅ All fixes verified!")
    else:
        print("❌ Some fixes need attention")
        print("\nRecommendations:")
        print("1. For negative rewards: Increase gate weight in training config")
        print("2. For excessive tools: Adjust reward weights to make penalties more impactful")
        print("   Example: reward_weights={'task': 0.5, 'tool': 0.1, 'gate': 0.4}")

if __name__ == "__main__":
    main()