# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Confidence tracking utilities for Arc Vision RL.

Since we're using VERL's multi-turn tool system, confidence is implicitly
handled through the model's decision to invoke tools. This module provides
utilities to analyze tool usage patterns as a proxy for confidence.
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np


def extract_tool_usage_as_confidence_proxy(response: str) -> Dict[str, float]:
    """Extract tool usage patterns as a proxy for model confidence.
    
    In Arc Vision RL, the model's decision to use tools indicates low confidence.
    We can infer confidence levels from:
    1. Whether tools were used
    2. Which tools were used
    3. Tool invocation patterns
    
    Args:
        response: Model's complete response including tool calls
        
    Returns:
        Dictionary with confidence metrics
    """
    confidence_metrics = {
        "implied_confidence": 1.0,  # Default high confidence
        "tool_invocations": 0,
        "tools_used": [],
        "confidence_before": 0.8,  # Estimated
        "confidence_after": 0.8    # Estimated
    }
    
    # Check for tool calls
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_calls = re.findall(tool_call_pattern, response, re.DOTALL)
    
    if tool_calls:
        confidence_metrics["tool_invocations"] = len(tool_calls)
        
        # Parse tool names
        for call in tool_calls:
            try:
                # Simple extraction of tool name
                if "zoom_ui_element" in call:
                    confidence_metrics["tools_used"].append("zoom")
                elif "wait_for_ui" in call:
                    confidence_metrics["tools_used"].append("wait")
                elif "inspect_element" in call:
                    confidence_metrics["tools_used"].append("inspect")
            except:
                pass
        
        # Infer confidence based on tool usage
        # More tools = lower initial confidence
        confidence_metrics["implied_confidence"] = max(0.3, 1.0 - 0.2 * len(tool_calls))
        confidence_metrics["confidence_before"] = confidence_metrics["implied_confidence"]
        
        # After tools, confidence should increase
        confidence_metrics["confidence_after"] = min(0.95, 
            confidence_metrics["confidence_before"] + 0.15 * len(tool_calls))
    
    # Check for explicit confidence statements
    conf_pattern = r'confidence[:\s]+(\d+\.?\d*)'
    conf_match = re.search(conf_pattern, response.lower())
    if conf_match:
        try:
            stated_conf = float(conf_match.group(1))
            # If confidence is stated as percentage
            if stated_conf > 1:
                stated_conf = stated_conf / 100
            confidence_metrics["stated_confidence"] = stated_conf
        except:
            pass
    
    return confidence_metrics


def analyze_reasoning_for_confidence(response: str) -> float:
    """Analyze the reasoning section to estimate confidence.
    
    Look for uncertainty indicators in the reasoning section.
    
    Args:
        response: Model response with reasoning
        
    Returns:
        Estimated confidence score
    """
    # Extract reasoning section
    reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
    reasoning_match = re.search(reasoning_pattern, response, re.DOTALL)
    
    if not reasoning_match:
        return 0.7  # Default medium confidence
    
    reasoning = reasoning_match.group(1).lower()
    
    # Uncertainty indicators
    low_confidence_phrases = [
        "unclear", "difficult to see", "partially obscured", "hard to locate",
        "not sure", "might be", "possibly", "blurry", "small", "low contrast",
        "challenging", "need tools", "can't see clearly", "uncertain"
    ]
    
    # Confidence indicators
    high_confidence_phrases = [
        "clearly visible", "easy to see", "obvious", "prominent",
        "definitely", "certain", "sure", "clear", "visible"
    ]
    
    # Count indicators
    low_conf_count = sum(1 for phrase in low_confidence_phrases if phrase in reasoning)
    high_conf_count = sum(1 for phrase in high_confidence_phrases if phrase in reasoning)
    
    # Calculate confidence based on indicator balance
    if low_conf_count + high_conf_count == 0:
        return 0.7  # Neutral
    
    confidence = (high_conf_count - low_conf_count) / (high_conf_count + low_conf_count)
    # Map from [-1, 1] to [0.3, 0.95]
    confidence = 0.625 + 0.325 * confidence
    
    return float(np.clip(confidence, 0.3, 0.95))


def compute_effective_confidence(response: str) -> Tuple[float, float]:
    """Compute effective confidence before and after tool use.
    
    This combines multiple signals to estimate confidence levels.
    
    Args:
        response: Complete model response
        
    Returns:
        Tuple of (confidence_before, confidence_after)
    """
    # Get tool usage metrics
    tool_metrics = extract_tool_usage_as_confidence_proxy(response)
    
    # Analyze reasoning
    reasoning_confidence = analyze_reasoning_for_confidence(response)
    
    # Combine signals
    if tool_metrics["tool_invocations"] > 0:
        # Tools were used - low initial confidence
        confidence_before = min(tool_metrics["confidence_before"], reasoning_confidence)
        confidence_after = tool_metrics["confidence_after"]
    else:
        # No tools used - high confidence throughout
        confidence_before = max(reasoning_confidence, 0.7)
        confidence_after = confidence_before
    
    # Use stated confidence if available
    if "stated_confidence" in tool_metrics:
        confidence_after = tool_metrics["stated_confidence"]
    
    return float(confidence_before), float(confidence_after)


def should_use_tools(confidence: float, threshold: float = 0.7) -> bool:
    """Determine if tools should be used based on confidence.
    
    Args:
        confidence: Current confidence level
        threshold: Confidence threshold for tool use
        
    Returns:
        Whether tools should be invoked
    """
    return confidence < threshold