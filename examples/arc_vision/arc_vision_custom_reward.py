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
"""Custom reward function for Arc Vision RL training."""

import json
import logging
import time
import os
import re
from typing import Dict, List, Any
from pathlib import Path

# Remove unused import - ArcVisionRewardScore not needed

# Add project root to sys.path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from examples.arc_vision.utils.confidence_tracker import (
    extract_tool_usage_as_confidence_proxy,
    analyze_reasoning_for_confidence,
    compute_effective_confidence
)

logger = logging.getLogger(__name__)


# ==============================================================================
# DETAILED LOGGING SYSTEM FOR ARC VISION RL MONITORING
# ==============================================================================

def setup_detailed_logging(output_dir: str = "outputs/arc_vision") -> Dict[str, str]:
    """Setup detailed logging directories and return file paths.
    
    TODO: Called once at training start to setup logging infrastructure
    """
    base_dir = Path(output_dir) / "detailed_logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    log_files = {
        "reasoning_traces": str(base_dir / "reasoning_traces.jsonl"),
        "confidence_calibration": str(base_dir / "confidence_calibration.jsonl"),
        "tool_patterns": str(base_dir / "tool_patterns.jsonl"),
        "contradictions": str(base_dir / "contradictions.jsonl")
    }
    
    # Initialize log files with headers
    for log_type, file_path in log_files.items():
        if not Path(file_path).exists():
            with open(file_path, 'w') as f:
                f.write(f"# {log_type.upper()} LOG - Arc Vision RL\n")
    
    logger.info(f"Detailed logging setup complete: {base_dir}")
    return log_files


def extract_reasoning_section(response: str) -> str:
    """Extract reasoning section from model response.
    
    TODO: Parses <reasoning>...</reasoning> tags from model output
    """
    reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
    match = re.search(reasoning_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback to <think>...</think> tags
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""


def log_reasoning_trace(prompt_str: str, response_str: str, actual_iou: float, 
                       ground_truth: List[float], log_file: str) -> None:
    """Log detailed reasoning traces for analysis.
    
    TODO: Captures reasoning traces to identify listener disagreement patterns
    """
    try:
        # Extract reasoning and tool information
        reasoning = extract_reasoning_section(response_str)
        tool_metrics = extract_tool_usage_as_confidence_proxy(response_str)
        confidence_before, confidence_after = compute_effective_confidence(response_str)
        
        trace_data = {
            "timestamp": time.time(),
            "prompt_length": len(prompt_str),
            "response_length": len(response_str),
            "reasoning_text": reasoning,
            "reasoning_length": len(reasoning),
            "tools_used": tool_metrics["tools_used"],
            "tool_invocations": tool_metrics["tool_invocations"],
            "confidence_before": confidence_before,
            "confidence_after": confidence_after,
            "implied_confidence": tool_metrics["implied_confidence"],
            "actual_iou": actual_iou,
            "ground_truth_bbox": ground_truth,
            # TODO: Add bbox parsing from response for complete analysis
            "has_reasoning": len(reasoning) > 0,
            "has_tool_calls": tool_metrics["tool_invocations"] > 0
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(trace_data) + '\n')
            
    except Exception as e:
        logger.warning(f"Failed to log reasoning trace: {e}")


def track_confidence_calibration(predicted_confidence: float, actual_iou: float, 
                                tool_used: bool, response_str: str, log_file: str) -> None:
    """Track how well confidence predicts actual performance.
    
    TODO: Monitors confidence calibration accuracy for model reliability
    """
    try:
        # Calculate calibration metrics
        calibration_error = abs(predicted_confidence - actual_iou)
        overconfident = predicted_confidence > actual_iou + 0.1
        underconfident = predicted_confidence < actual_iou - 0.1
        
        # Analyze reasoning confidence indicators
        reasoning_confidence = analyze_reasoning_for_confidence(response_str)
        
        calibration_data = {
            "timestamp": time.time(),
            "predicted_confidence": predicted_confidence,
            "actual_iou": actual_iou,
            "calibration_error": calibration_error,
            "overconfident": overconfident,
            "underconfident": underconfident,
            "tool_used": tool_used,
            "reasoning_confidence": reasoning_confidence,
            "confidence_alignment": abs(predicted_confidence - reasoning_confidence),
            # Performance categories for analysis
            "performance_category": (
                "excellent" if actual_iou > 0.8 else
                "good" if actual_iou > 0.5 else
                "poor" if actual_iou > 0.1 else
                "failed"
            ),
            "confidence_category": (
                "high" if predicted_confidence > 0.8 else
                "medium" if predicted_confidence > 0.5 else
                "low"
            )
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(calibration_data) + '\n')
            
    except Exception as e:
        logger.warning(f"Failed to track confidence calibration: {e}")


def monitor_tool_patterns(response_str: str, actual_iou: float, 
                         ground_truth: List[float], log_file: str) -> None:
    """Monitor tool usage patterns for effectiveness analysis.
    
    TODO: Analyzes tool selection patterns and effectiveness
    """
    try:
        tool_metrics = extract_tool_usage_as_confidence_proxy(response_str)
        confidence_before, confidence_after = compute_effective_confidence(response_str)
        
        # Analyze tool effectiveness
        tool_effective = (
            tool_metrics["tool_invocations"] > 0 and 
            confidence_after > confidence_before + 0.1
        )
        
        # Calculate confidence gain from tools
        confidence_gain = confidence_after - confidence_before if tool_metrics["tool_invocations"] > 0 else 0.0
        
        pattern_data = {
            "timestamp": time.time(),
            "tools_used": tool_metrics["tools_used"],
            "tool_count": tool_metrics["tool_invocations"],
            "confidence_before": confidence_before,
            "confidence_after": confidence_after,
            "confidence_gain": confidence_gain,
            "actual_iou": actual_iou,
            "tool_effective": tool_effective,
            "ground_truth_area": (
                float(ground_truth[2]) - float(ground_truth[0])
            ) * (
                float(ground_truth[3]) - float(ground_truth[1])
            ) if isinstance(ground_truth, (list, tuple)) and len(ground_truth) >= 4 else 0.0,
            # Tool usage analysis
            "used_zoom": "zoom" in tool_metrics["tools_used"],
            "used_wait": "wait" in tool_metrics["tools_used"],
            "used_inspect": "inspect" in tool_metrics["tools_used"],
            "multiple_tools": len(tool_metrics["tools_used"]) > 1,
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(pattern_data) + '\n')
            
    except Exception as e:
        logger.warning(f"Failed to monitor tool patterns: {e}")


def detect_tool_contradictions(response_str: str, actual_iou: float, 
                              ground_truth: List[float], log_file: str) -> None:
    """Detect contradictions between tool decisions and outcomes.
    
    TODO: Identifies listener disagreement patterns and reasoning contradictions
    """
    try:
        tool_metrics = extract_tool_usage_as_confidence_proxy(response_str)
        confidence_before, confidence_after = compute_effective_confidence(response_str)
        reasoning = extract_reasoning_section(response_str)
        
        # Detect various contradiction patterns
        contradictions = {
            # Tool used but high performance without need
            "unnecessary_tool_use": (
                tool_metrics["tool_invocations"] > 0 and 
                actual_iou > 0.8
            ),
            # No tool used but poor performance suggests need
            "missed_tool_opportunity": (
                tool_metrics["tool_invocations"] == 0 and 
                actual_iou < 0.3 and
                confidence_before > 0.6  # Overconfident
            ),
            # Tool used but no confidence improvement
            "ineffective_tool_use": (
                tool_metrics["tool_invocations"] > 0 and
                (confidence_after - confidence_before) < 0.05
            ),
            # High confidence but poor performance
            "overconfidence_failure": (
                confidence_before > 0.8 and
                actual_iou < 0.3
            ),
            # Low confidence but good performance
            "underconfidence_success": (
                confidence_before < 0.4 and
                actual_iou > 0.7
            ),
            # Reasoning suggests uncertainty but no tools used
            "reasoning_tool_mismatch": (
                len(reasoning) > 0 and
                any(phrase in reasoning.lower() for phrase in 
                    ["unclear", "difficult", "hard to see", "not sure"]) and
                tool_metrics["tool_invocations"] == 0
            )
        }
        
        # Only log if contradictions detected
        if any(contradictions.values()):
            contradiction_data = {
                "timestamp": time.time(),
                "actual_iou": actual_iou,
                "confidence_before": confidence_before,
                "confidence_after": confidence_after,
                "tools_used": tool_metrics["tools_used"],
                "tool_count": tool_metrics["tool_invocations"],
                "reasoning_length": len(reasoning),
                "contradictions": contradictions,
                "contradiction_count": sum(contradictions.values()),
                "reasoning_snippet": reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(contradiction_data) + '\n')
                
    except Exception as e:
        logger.warning(f"Failed to detect contradictions: {e}")


# Global variable to store log file paths (initialized once)
_LOG_FILES = None


def get_log_files() -> Dict[str, str]:
    """Get or initialize log file paths."""
    global _LOG_FILES
    if _LOG_FILES is None:
        _LOG_FILES = setup_detailed_logging()
    return _LOG_FILES


def arc_vision_compute_reward(data_source: str, 
                            solution_str: str, 
                            ground_truth: Any, 
                            extra_info: Dict = None,
                            confidence_threshold: float = 0.7,
                            reward_weights: Dict[str, float] = None,
                            tool_penalties: Dict[str, float] = None,
                            **kwargs):
    """Custom reward function for Arc Vision that integrates with VERL's reward manager.
    
    Implements the complete 3-component reward function from the blog post:
    R(s,a,t) = α*R_task + β*R_tool + γ*R_gate
    
    Where:
    - R_task: IoU-based detection accuracy
    - R_tool: Tool effectiveness based on confidence gain
    - R_gate: Penalties for tool misuse
    
    Args:
        data_source: Dataset identifier (should be "arc_vision" or "screenspot")
        solution_str: Model response string containing reasoning and tool usage
        ground_truth: Ground truth bounding box coordinates [x1, y1, x2, y2]
        extra_info: Additional information (unused for Arc Vision)
        confidence_threshold: Threshold for confidence-gated tool invocation (default: 0.7)
        reward_weights: Weights for reward components (default: α=0.6, β=0.3, γ=0.1)
        tool_penalties: Penalties for different tool usage failure modes
        **kwargs: Additional keyword arguments
        
    Returns:
        Dict: Reward score with detailed metrics
    """
    # Verify this is an Arc Vision request
    # Handle case where data_source might be passed as array/list
    if hasattr(data_source, '__len__') and not isinstance(data_source, str):
        # If it's an array/list, take the first element
        data_source = data_source[0] if len(data_source) > 0 else "unknown"
    
    if data_source not in ["arc_vision", "screenspot", "rootsautomation/ScreenSpot"]:
        logger.warning(f"Arc Vision reward function called with data_source: {data_source}")
    
    # Handle JSON string extra_info (from parquet storage)
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except (json.JSONDecodeError, TypeError):
            extra_info = {}
    
    # Use default parameters from blog post if not provided
    if reward_weights is None:
        reward_weights = {"task": 0.6, "tool": 0.3, "gate": 0.1}
    if tool_penalties is None:
        tool_penalties = {
            "unnecessary_tool": -0.5,
            "missed_opportunity": -0.3,
            "ineffective_tool": -0.2,
            "excessive_tools": -0.4
        }
    
    # Initialize reward components
    r_task = 0.0
    r_tool = 0.0
    r_gate = 0.0
    
    # ==============================================================================
    # 1. R_task: IoU-based Detection Accuracy
    # ==============================================================================
    import re
    
    # Extract predicted bbox from solution
    bbox_pattern = r'<bbox>\s*\[([\d\.\s,]+)\]\s*</bbox>'
    match = re.search(bbox_pattern, solution_str, re.IGNORECASE)
    
    predicted_bbox = None
    iou = 0.0
    
    if match:
        try:
            predicted_bbox = [float(x.strip()) for x in match.group(1).split(',')]
            
            # Calculate IoU
            x1 = max(predicted_bbox[0], ground_truth[0])
            y1 = max(predicted_bbox[1], ground_truth[1])
            x2 = min(predicted_bbox[2], ground_truth[2])
            y2 = min(predicted_bbox[3], ground_truth[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area_pred = (predicted_bbox[2] - predicted_bbox[0]) * (predicted_bbox[3] - predicted_bbox[1])
                area_gt = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])
                union = area_pred + area_gt - intersection
                iou = intersection / union if union > 0 else 0.0
            else:
                iou = 0.0
            
            r_task = iou
        except Exception as e:
            logger.warning(f"Failed to parse bbox: {e}")
            r_task = 0.0
    else:
        r_task = 0.0
    
    # ==============================================================================
    # 2. Extract Tool Usage and Confidence Information
    # ==============================================================================
    tool_metrics = extract_tool_usage_as_confidence_proxy(solution_str)
    confidence_before, confidence_after = compute_effective_confidence(solution_str)
    
    # ==============================================================================
    # 3. R_tool: Tool Effectiveness Based on Confidence Gain
    # ==============================================================================
    if tool_metrics["tool_invocations"] > 0:
        # Calculate confidence gain from tool use
        confidence_gain = confidence_after - confidence_before
        
        # R_tool = confidence_gain * IoU (tool effectiveness tied to task success)
        r_tool = max(0.0, confidence_gain * iou)
        
        # Cap maximum tool reward
        r_tool = min(r_tool, 1.0)
    else:
        # No tools used, no tool reward
        r_tool = 0.0
    
    # ==============================================================================
    # 4. R_gate: Penalties for Tool Misuse
    # ==============================================================================
    gate_penalties = []
    
    # Penalty 1: Unnecessary tool use (high confidence but used tools)
    if confidence_before >= confidence_threshold and tool_metrics["tool_invocations"] > 0:
        penalty = tool_penalties["unnecessary_tool"]
        gate_penalties.append(("unnecessary_tool", penalty))
    
    # Penalty 2: Missed opportunity (low confidence but no tools)
    if confidence_before < confidence_threshold and tool_metrics["tool_invocations"] == 0:
        penalty = tool_penalties["missed_opportunity"]
        gate_penalties.append(("missed_opportunity", penalty))
    
    # Penalty 3: Ineffective tool use (tool used but no confidence gain)
    if tool_metrics["tool_invocations"] > 0:
        confidence_gain = confidence_after - confidence_before
        if confidence_gain < 0.05:  # Less than 5% confidence gain
            penalty = tool_penalties["ineffective_tool"]
            gate_penalties.append(("ineffective_tool", penalty))
    
    # Penalty 4: Excessive tool use (more than 2 tools)
    # Note: max_assistant_turns=2 in our config, so this triggers for >2 tools
    if tool_metrics["tool_invocations"] > 2:
        penalty = tool_penalties["excessive_tools"]
        gate_penalties.append(("excessive_tools", penalty))
    
    # Sum all gate penalties (they are negative values)
    r_gate = sum(penalty for _, penalty in gate_penalties)
    
    # ==============================================================================
    # 5. Compute Final Reward: R(s,a,t) = α*R_task + β*R_tool + γ*R_gate
    # ==============================================================================
    final_reward = (
        reward_weights["task"] * r_task + 
        reward_weights["tool"] * r_tool + 
        reward_weights["gate"] * r_gate
    )
    
    # Do not clamp - allow negative rewards to propagate for proper RL signal
    
    # ==============================================================================
    # DETAILED LOGGING INTEGRATION - Log all monitoring data
    # ==============================================================================
    try:
        log_files = get_log_files()
        
        # 1. Log reasoning traces - captures reasoning for listener analysis
        log_reasoning_trace(
            prompt_str="",  # Not available in this interface
            response_str=solution_str,
            actual_iou=iou,
            ground_truth=ground_truth,
            log_file=log_files["reasoning_traces"]
        )
        
        # 2. Track confidence calibration - monitors prediction accuracy
        track_confidence_calibration(
            predicted_confidence=confidence_before,
            actual_iou=iou,
            tool_used=tool_metrics["tool_invocations"] > 0,
            response_str=solution_str,
            log_file=log_files["confidence_calibration"]
        )
        
        # 3. Monitor tool patterns - analyzes tool effectiveness
        monitor_tool_patterns(
            response_str=solution_str,
            actual_iou=iou,
            ground_truth=ground_truth,
            log_file=log_files["tool_patterns"]
        )
        
        # 4. Detect contradictions - identifies listener disagreement patterns
        detect_tool_contradictions(
            response_str=solution_str,
            actual_iou=iou,
            ground_truth=ground_truth,
            log_file=log_files["contradictions"]
        )
        
    except Exception as e:
        # Don't fail training if logging fails
        logger.warning(f"Detailed logging failed: {e}")
    
    # Log reward statistics for debugging
    logger.info(f"Arc Vision reward breakdown - Task: {r_task:.3f}, Tool: {r_tool:.3f}, Gate: {r_gate:.3f}")
    logger.info(f"Confidence: {confidence_before:.3f} -> {confidence_after:.3f}, Tools used: {tool_metrics['tool_invocations']}")
    logger.info(f"Final reward: {final_reward:.3f} (IoU: {iou:.3f})")
    
    # Return detailed reward information
    # VERL's reward manager expects a dict with at least a 'score' key
    return {
        "score": float(final_reward),
        # Additional metrics for analysis
        "r_task": float(r_task),
        "r_tool": float(r_tool),
        "r_gate": float(r_gate),
        "iou": float(iou),
        "confidence_before": float(confidence_before),
        "confidence_after": float(confidence_after),
        "tool_invocations": tool_metrics["tool_invocations"],
        "tools_used": tool_metrics["tools_used"],
        "gate_penalties": gate_penalties,
        "predicted_bbox": predicted_bbox,
        "ground_truth": ground_truth
    }


def create_arc_vision_compute_score_fn(confidence_threshold: float = 0.7,
                                      reward_weights: Dict[str, float] = None,
                                      tool_penalties: Dict[str, float] = None):
    """Create a compute_score function configured for Arc Vision.
    
    This factory function creates a compute_score function with the Arc Vision
    parameters pre-configured. This allows VERL's reward manager to use it
    directly without needing to pass custom parameters.
    
    Args:
        confidence_threshold: Threshold for confidence-gated tool invocation
        reward_weights: Weights for reward components (task, tool, gate)
        tool_penalties: Penalties for different tool usage failure modes
        
    Returns:
        Function that matches VERL's compute_score interface
    """
    def compute_score(data_source: str, solution_str: str, ground_truth: Any, 
                     extra_info: Dict = None, **kwargs) -> Dict:
        return arc_vision_compute_reward(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            confidence_threshold=confidence_threshold,
            reward_weights=reward_weights,
            tool_penalties=tool_penalties,
            **kwargs
        )
    
    return compute_score


# Create the default Arc Vision compute score function
arc_vision_compute_score_fn = create_arc_vision_compute_score_fn(
    confidence_threshold=0.7,
    reward_weights={"task": 0.6, "tool": 0.3, "gate": 0.1},
    tool_penalties={
        "unnecessary_tool": -0.5,
        "missed_opportunity": -0.3,
        "ineffective_tool": -0.2,
        "excessive_tools": -0.4
    }
)