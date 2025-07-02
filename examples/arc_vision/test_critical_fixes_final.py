#!/usr/bin/env python3
"""Final test of the two critical fixes with proper test cases."""

import sys
from pathlib import Path

# Add the examples directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_vision.arc_vision_custom_reward import arc_vision_compute_reward

def test_negative_rewards_with_tools():
    """Test negative rewards through ineffective tool use."""
    print("=== Testing Negative Rewards (via ineffective tools) ===\n")
    
    # Create a response with tools but no confidence gain
    response = """<reasoning>
I can't see the element clearly, trying tools.
</reasoning>

<tool_call>
name: zoom
keypoint: [0.5, 0.5]
</tool_call>

<bbox>[0.1, 0.1, 0.2, 0.2]</bbox>"""
    
    result = arc_vision_compute_reward(
        data_source="arc_vision",
        solution_str=response,
        ground_truth=[0.8, 0.8, 0.9, 0.9],  # Far from prediction
        extra_info={"index": 0}
    )
    
    print("Test case: Poor detection with ineffective tool use")
    print(f"IoU: {result['iou']:.3f} (no overlap)")
    print(f"R_task: {result['r_task']:.3f}")
    print(f"R_tool: {result['r_tool']:.3f}")
    print(f"R_gate: {result['r_gate']:.3f}")
    print(f"Gate penalties: {result['gate_penalties']}")
    print(f"Final reward: {result['reward']:.3f}")
    
    # With default weights: task=0.6, tool=0.3, gate=0.1
    # R_task = 0.0 (no IoU)
    # R_tool = 0.0 (no confidence gain * IoU = 0)
    # R_gate = -0.2 (ineffective_tool penalty)
    # Final = 0.6*0.0 + 0.3*0.0 + 0.1*(-0.2) = -0.02
    
    if result['reward'] < 0:
        print("✅ Negative reward successfully propagated!\n")
        return True
    else:
        print("❌ ERROR: Reward should be negative\n")
        return False

def test_negative_rewards_high_confidence_with_tools():
    """Test negative rewards through unnecessary tool use."""
    print("=== Testing Negative Rewards (unnecessary tools) ===\n")
    
    # Create a response that indicates high confidence but uses tools anyway
    response = """<reasoning>
The element is clearly visible and easy to locate. I can see it perfectly.
Let me zoom in anyway to be extra sure.
</reasoning>

<tool_call>
name: zoom
keypoint: [0.5, 0.5]
</tool_call>

<bbox>[0.5, 0.5, 0.6, 0.6]</bbox>"""
    
    result = arc_vision_compute_reward(
        data_source="arc_vision",
        solution_str=response,
        ground_truth=[0.5, 0.5, 0.6, 0.6],  # Perfect match
        extra_info={"index": 0},
        confidence_threshold=0.7
    )
    
    print("Test case: Perfect detection but unnecessary tool use")
    print(f"IoU: {result['iou']:.3f}")
    print(f"Confidence before: {result['confidence_before']:.3f}")
    print(f"R_task: {result['r_task']:.3f}")
    print(f"R_tool: {result['r_tool']:.3f}")
    print(f"R_gate: {result['r_gate']:.3f}")
    print(f"Gate penalties: {result['gate_penalties']}")
    print(f"Final reward: {result['reward']:.3f}")
    
    # Even with perfect IoU, unnecessary tool penalty should reduce reward
    has_penalty = any(p[0] == "unnecessary_tool" for p in result['gate_penalties'])
    
    if has_penalty:
        print("✅ Unnecessary tool penalty applied!\n")
        return True
    else:
        print("❌ ERROR: Should have unnecessary tool penalty\n")
        return False

def test_excessive_tools_comprehensive():
    """Comprehensive test of excessive tools penalty."""
    print("=== Testing Excessive Tools Penalty ===\n")
    
    test_cases = [
        (1, """<tool_call>
name: zoom
keypoint: [0.5, 0.5]
</tool_call>

<bbox>[0.5, 0.5, 0.6, 0.6]</bbox>"""),
        (2, """<tool_call>
name: zoom
keypoint: [0.5, 0.5]
</tool_call>

<tool_call>
name: wait
duration: 1
</tool_call>

<bbox>[0.5, 0.5, 0.6, 0.6]</bbox>"""),
        (3, """<tool_call>
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

<bbox>[0.5, 0.5, 0.6, 0.6]</bbox>"""),
        (4, """<tool_call>
name: zoom
keypoint: [0.4, 0.4]
</tool_call>

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

<bbox>[0.5, 0.5, 0.6, 0.6]</bbox>""")
    ]
    
    results = []
    for num_tools, response in test_cases:
        result = arc_vision_compute_reward(
            data_source="arc_vision",
            solution_str=response,
            ground_truth=[0.5, 0.5, 0.6, 0.6],
            extra_info={"index": num_tools}
        )
        
        has_excessive = any(p[0] == "excessive_tools" for p in result['gate_penalties'])
        results.append((num_tools, result['reward'], has_excessive))
        
        print(f"{num_tools} tool(s): reward={result['reward']:.3f}, excessive_penalty={has_excessive}")
    
    # Check that penalty triggers at >2
    correct_triggers = all([
        not results[0][2],  # 1 tool: no penalty
        not results[1][2],  # 2 tools: no penalty
        results[2][2],      # 3 tools: penalty
        results[3][2]       # 4 tools: penalty
    ])
    
    if correct_triggers:
        print("\n✅ Excessive tools penalty triggers correctly at >2 tools!")
        return True
    else:
        print("\n❌ ERROR: Excessive tools penalty not triggering correctly")
        return False

def main():
    print("=== Final Validation of Critical Fixes ===\n")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Negative rewards via ineffective tools
    if test_negative_rewards_with_tools():
        tests_passed += 1
    
    # Test 2: Negative rewards via unnecessary tools
    if test_negative_rewards_high_confidence_with_tools():
        tests_passed += 1
    
    # Test 3: Excessive tools penalty
    if test_excessive_tools_comprehensive():
        tests_passed += 1
    
    print(f"\n{'='*60}")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n✅ ALL CRITICAL FIXES VALIDATED!")
        print("\nSummary:")
        print("1. ✅ Reward clamping removed - negative rewards can propagate")
        print("2. ✅ Excessive tools penalty triggers at >2 tools")
        print("\nThe reward function is ready for training without dead code paths.")
        return True
    else:
        print("\n❌ Some fixes still need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)