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

import torch

from verl import DataProto
from verl.utils.reward_score.arc_vision_reward import ArcVisionRewardScore
from verl.examples.arc_vision.utils.confidence_tracker import (
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
            "ground_truth_area": (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1]) if len(ground_truth) >= 4 else 0.0,
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


def arc_vision_compute_reward(data: DataProto, 
                            return_dict: bool = False,
                            confidence_threshold: float = 0.7,
                            reward_weights: Dict[str, float] = None,
                            tool_penalties: Dict[str, float] = None,
                            **kwargs):
    """Custom reward function for Arc Vision that integrates with VERL's PPO trainer.
    
    This function is called by VERL's reward manager to compute rewards for
    Arc Vision responses that include tool usage for UI element detection.
    
    Args:
        data: DataProto containing prompts and responses
        confidence_threshold: Threshold for confidence-gated tool invocation
        reward_weights: Weights for reward components (task, tool, gate)
        tool_penalties: Penalties for different tool usage failure modes
        **kwargs: Additional keyword arguments
        
    Returns:
        torch.Tensor: Reward scores for each response in the batch
    """
    # Initialize Arc Vision reward model
    reward_model = ArcVisionRewardScore(
        confidence_threshold=confidence_threshold,
        reward_weights=reward_weights,
        tool_penalties=tool_penalties
    )
    
    rewards = []
    
    # Process each item in the batch
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem
        
        # Extract prompt and response strings
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        
        # Decode to strings (tokenizer passed via kwargs)
        tokenizer = kwargs.get("tokenizer")
        if tokenizer:
            prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        else:
            # Fallback to raw strings if available
            prompt_str = data_item.non_tensor_batch.get("raw_prompt", "")
            response_str = data_item.non_tensor_batch.get("response", "")
        
        # Extract ground truth from reward_model dict
        reward_model_data = data_item.non_tensor_batch.get("reward_model", {})
        ground_truth = reward_model_data.get("ground_truth", None)
        
        if ground_truth is None:
            # No ground truth available for this sample
            rewards.append(0.0)
            continue
        
        # Prepare ground truth data structure
        gt_data = {"ground_truth": ground_truth}
        
        # Compute reward using Arc Vision reward model
        reward_list = reward_model(
            questions=[prompt_str],
            responses=[response_str],
            reward_model=[gt_data]
        )
        current_reward = reward_list[0]
        rewards.append(current_reward)
        
        # ==============================================================================
        # DETAILED LOGGING INTEGRATION - Log all monitoring data
        # ==============================================================================
        try:
            log_files = get_log_files()
            
            # Extract IoU from the reward model for logging
            # TODO: Access actual IoU computation from reward model
            # For now, use reward as proxy (will be enhanced in reward model)
            actual_iou = max(0.0, min(1.0, current_reward))  # Clamp to [0,1]
            
            # Extract confidence estimates
            confidence_before, confidence_after = compute_effective_confidence(response_str)
            tool_metrics = extract_tool_usage_as_confidence_proxy(response_str)
            
            # 1. Log reasoning traces - captures reasoning for listener analysis
            log_reasoning_trace(
                prompt_str=prompt_str,
                response_str=response_str,
                actual_iou=actual_iou,
                ground_truth=ground_truth,
                log_file=log_files["reasoning_traces"]
            )
            
            # 2. Track confidence calibration - monitors prediction accuracy
            track_confidence_calibration(
                predicted_confidence=confidence_before,
                actual_iou=actual_iou,
                tool_used=tool_metrics["tool_invocations"] > 0,
                response_str=response_str,
                log_file=log_files["confidence_calibration"]
            )
            
            # 3. Monitor tool patterns - analyzes tool effectiveness
            monitor_tool_patterns(
                response_str=response_str,
                actual_iou=actual_iou,
                ground_truth=ground_truth,
                log_file=log_files["tool_patterns"]
            )
            
            # 4. Detect contradictions - identifies listener disagreement patterns
            detect_tool_contradictions(
                response_str=response_str,
                actual_iou=actual_iou,
                ground_truth=ground_truth,
                log_file=log_files["contradictions"]
            )
            
        except Exception as e:
            # Don't fail training if logging fails
            logger.warning(f"Detailed logging failed for sample {i}: {e}")
    
    # Convert to tensor
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    
    # Log some statistics for debugging
    if len(rewards) > 0:
        logger.info(f"Arc Vision rewards - Mean: {reward_tensor.mean():.3f}, "
                   f"Std: {reward_tensor.std():.3f}, "
                   f"Min: {reward_tensor.min():.3f}, "
                   f"Max: {reward_tensor.max():.3f}")
    
    if return_dict:
        return {"reward_tensor": reward_tensor}
    else:
        return reward_tensor