#!/usr/bin/env python3
"""Test script for Arc Vision reward function implementation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.arc_vision.arc_vision_custom_reward import arc_vision_compute_score_fn


def test_case_1_perfect_detection_no_tools():
    """Test case: Perfect detection without tools (high confidence)."""
    print("\n=== Test Case 1: Perfect Detection, No Tools ===")
    
    response = """<reasoning>
    The button is clearly visible in the center of the screen. It's prominent and easy to locate.
    </reasoning>
    
    <bbox>[100, 200, 300, 400]</bbox>"""
    
    ground_truth = [100, 200, 300, 400]
    
    result = arc_vision_compute_score_fn(
        data_source="arc_vision",
        solution_str=response,
        ground_truth=ground_truth
    )
    
    print(f"Response: High confidence, no tools used")
    print(f"Ground truth bbox: {ground_truth}")
    print(f"Predicted bbox: {result['predicted_bbox']}")
    print(f"IoU: {result['iou']:.3f}")
    print(f"Confidence: {result['confidence_before']:.3f} -> {result['confidence_after']:.3f}")
    print(f"Tools used: {result['tool_invocations']}")
    print(f"Reward breakdown:")
    print(f"  R_task: {result['r_task']:.3f}")
    print(f"  R_tool: {result['r_tool']:.3f}")
    print(f"  R_gate: {result['r_gate']:.3f}")
    print(f"  Total: {result['reward']:.3f}")
    print(f"Gate penalties: {result['gate_penalties']}")
    
    expected_reward = 0.6 * 1.0 + 0.3 * 0.0 + 0.1 * 0.0  # 0.6
    assert abs(result['reward'] - expected_reward) < 0.01, f"Expected reward ~{expected_reward}, got {result['reward']}"


def test_case_2_low_confidence_with_tools():
    """Test case: Low confidence, uses tools effectively."""
    print("\n=== Test Case 2: Low Confidence, Effective Tool Use ===")
    
    response = """<reasoning>
    The element is difficult to see clearly. It appears to be partially obscured. I'll need to use tools to get a better view.
    </reasoning>
    
    <tool_call>
        <name>zoom_ui_element</name>
        <parameters>region=[50, 150, 350, 450]</parameters>
    </tool_call>
    
    After zooming in, I can now clearly see the button boundaries.
    
    <bbox>[100, 200, 300, 400]</bbox>"""
    
    ground_truth = [100, 200, 300, 400]
    
    result = arc_vision_compute_score_fn(
        data_source="arc_vision",
        solution_str=response,
        ground_truth=ground_truth
    )
    
    print(f"Response: Low confidence, tool used effectively")
    print(f"Ground truth bbox: {ground_truth}")
    print(f"Predicted bbox: {result['predicted_bbox']}")
    print(f"IoU: {result['iou']:.3f}")
    print(f"Confidence: {result['confidence_before']:.3f} -> {result['confidence_after']:.3f}")
    print(f"Tools used: {result['tools_used']}")
    print(f"Reward breakdown:")
    print(f"  R_task: {result['r_task']:.3f}")
    print(f"  R_tool: {result['r_tool']:.3f}")
    print(f"  R_gate: {result['r_gate']:.3f}")
    print(f"  Total: {result['reward']:.3f}")
    print(f"Gate penalties: {result['gate_penalties']}")


def test_case_3_unnecessary_tool_use():
    """Test case: High confidence but uses tools unnecessarily."""
    print("\n=== Test Case 3: Unnecessary Tool Use ===")
    
    response = """<reasoning>
    The button is clearly visible and prominent on the screen.
    </reasoning>
    
    <tool_call>
        <name>zoom_ui_element</name>
        <parameters>region=[50, 150, 350, 450]</parameters>
    </tool_call>
    
    <bbox>[100, 200, 300, 400]</bbox>"""
    
    ground_truth = [100, 200, 300, 400]
    
    result = arc_vision_compute_score_fn(
        data_source="arc_vision",
        solution_str=response,
        ground_truth=ground_truth
    )
    
    print(f"Response: High confidence but used tool")
    print(f"Ground truth bbox: {ground_truth}")
    print(f"Predicted bbox: {result['predicted_bbox']}")
    print(f"IoU: {result['iou']:.3f}")
    print(f"Confidence: {result['confidence_before']:.3f} -> {result['confidence_after']:.3f}")
    print(f"Tools used: {result['tools_used']}")
    print(f"Reward breakdown:")
    print(f"  R_task: {result['r_task']:.3f}")
    print(f"  R_tool: {result['r_tool']:.3f}")
    print(f"  R_gate: {result['r_gate']:.3f}")
    print(f"  Total: {result['reward']:.3f}")
    print(f"Gate penalties: {result['gate_penalties']}")


def test_case_4_missed_opportunity():
    """Test case: Low confidence but doesn't use tools."""
    print("\n=== Test Case 4: Missed Tool Opportunity ===")
    
    response = """<reasoning>
    The element is very difficult to see. It's unclear and partially obscured. I'm not sure about the exact boundaries.
    </reasoning>
    
    <bbox>[120, 220, 280, 380]</bbox>"""
    
    ground_truth = [100, 200, 300, 400]
    
    result = arc_vision_compute_score_fn(
        data_source="arc_vision",
        solution_str=response,
        ground_truth=ground_truth
    )
    
    print(f"Response: Low confidence, no tools used")
    print(f"Ground truth bbox: {ground_truth}")
    print(f"Predicted bbox: {result['predicted_bbox']}")
    print(f"IoU: {result['iou']:.3f}")
    print(f"Confidence: {result['confidence_before']:.3f} -> {result['confidence_after']:.3f}")
    print(f"Tools used: {result['tool_invocations']}")
    print(f"Reward breakdown:")
    print(f"  R_task: {result['r_task']:.3f}")
    print(f"  R_tool: {result['r_tool']:.3f}")
    print(f"  R_gate: {result['r_gate']:.3f}")
    print(f"  Total: {result['reward']:.3f}")
    print(f"Gate penalties: {result['gate_penalties']}")


def test_case_5_excessive_tools():
    """Test case: Uses too many tools."""
    print("\n=== Test Case 5: Excessive Tool Use ===")
    
    response = """<reasoning>
    I need to examine this element more carefully.
    </reasoning>
    
    <tool_call>
        <name>zoom_ui_element</name>
        <parameters>region=[0, 0, 500, 500]</parameters>
    </tool_call>
    
    <tool_call>
        <name>wait_for_ui</name>
        <parameters>timeout=1000</parameters>
    </tool_call>
    
    <tool_call>
        <name>inspect_element</name>
        <parameters>element="button"</parameters>
    </tool_call>
    
    <tool_call>
        <name>zoom_ui_element</name>
        <parameters>region=[50, 150, 350, 450]</parameters>
    </tool_call>
    
    <bbox>[100, 200, 300, 400]</bbox>"""
    
    ground_truth = [100, 200, 300, 400]
    
    result = arc_vision_compute_score_fn(
        data_source="arc_vision",
        solution_str=response,
        ground_truth=ground_truth
    )
    
    print(f"Response: Used 4 tools (excessive)")
    print(f"Ground truth bbox: {ground_truth}")
    print(f"Predicted bbox: {result['predicted_bbox']}")
    print(f"IoU: {result['iou']:.3f}")
    print(f"Confidence: {result['confidence_before']:.3f} -> {result['confidence_after']:.3f}")
    print(f"Tools used: {result['tools_used']} (count: {result['tool_invocations']})")
    print(f"Reward breakdown:")
    print(f"  R_task: {result['r_task']:.3f}")
    print(f"  R_tool: {result['r_tool']:.3f}")
    print(f"  R_gate: {result['r_gate']:.3f}")
    print(f"  Total: {result['reward']:.3f}")
    print(f"Gate penalties: {result['gate_penalties']}")


def test_case_6_no_bbox_provided():
    """Test case: Model fails to provide bbox."""
    print("\n=== Test Case 6: No Bbox Provided ===")
    
    response = """<reasoning>
    I cannot locate the element in the image.
    </reasoning>"""
    
    ground_truth = [100, 200, 300, 400]
    
    result = arc_vision_compute_score_fn(
        data_source="arc_vision",
        solution_str=response,
        ground_truth=ground_truth
    )
    
    print(f"Response: No bbox provided")
    print(f"Ground truth bbox: {ground_truth}")
    print(f"Predicted bbox: {result['predicted_bbox']}")
    print(f"IoU: {result['iou']:.3f}")
    print(f"Confidence: {result['confidence_before']:.3f} -> {result['confidence_after']:.3f}")
    print(f"Tools used: {result['tool_invocations']}")
    print(f"Reward breakdown:")
    print(f"  R_task: {result['r_task']:.3f}")
    print(f"  R_tool: {result['r_tool']:.3f}")
    print(f"  R_gate: {result['r_gate']:.3f}")
    print(f"  Total: {result['reward']:.3f}")
    print(f"Gate penalties: {result['gate_penalties']}")


def main():
    """Run all test cases."""
    print("Arc Vision Reward Function Test Suite")
    print("=====================================")
    
    test_case_1_perfect_detection_no_tools()
    test_case_2_low_confidence_with_tools()
    test_case_3_unnecessary_tool_use()
    test_case_4_missed_opportunity()
    test_case_5_excessive_tools()
    test_case_6_no_bbox_provided()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()