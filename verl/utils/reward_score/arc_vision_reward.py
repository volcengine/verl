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
"""Arc Vision RL reward computation for UI element detection with tool learning."""

import json
import re
from typing import Dict, List, Any, Tuple
import numpy as np


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute Intersection over Union between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        IoU score between 0 and 1
    """
    # Ensure numpy arrays
    bbox1 = np.array(bbox1, dtype=np.float32)
    bbox2 = np.array(bbox2, dtype=np.float32)
    
    # Compute intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute areas
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Compute union
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union <= 0:
        return 0.0
    
    return float(intersection / union)


def parse_bbox_from_response(response: str) -> Tuple[np.ndarray, bool]:
    """Extract bounding box from model response.
    
    Args:
        response: Model's text response
        
    Returns:
        Tuple of (bbox array, success flag)
    """
    # Try multiple patterns for bbox extraction
    patterns = [
        r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]',  # [x1, y1, x2, y2]
        r'bbox:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]',  # bbox: [x1, y1, x2, y2]
        r'<bbox>\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]</bbox>',  # <bbox>[x1, y1, x2, y2]</bbox>
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                bbox = np.array([float(x) for x in match.groups()], dtype=np.float32)
                # Validate bbox (coordinates should be normalized 0-1)
                if np.all(bbox >= 0) and np.all(bbox <= 1) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    return bbox, True
            except:
                continue
    
    return np.array([0, 0, 0, 0], dtype=np.float32), False


def parse_tool_usage(response: str) -> Dict[str, Any]:
    """Extract tool usage information from response.
    
    Args:
        response: Model's text response
        
    Returns:
        Dictionary with tool usage info
    """
    tool_info = {
        "tool_used": False,
        "tool_name": None,
        "tool_calls": 0,
        "confidence_before": 0.5,
        "confidence_after": 0.5
    }
    
    # Check for tool calls
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_calls = re.findall(tool_call_pattern, response, re.DOTALL)
    
    if tool_calls:
        tool_info["tool_used"] = True
        tool_info["tool_calls"] = len(tool_calls)
        
        # Try to parse tool name from first call
        try:
            first_call = json.loads(tool_calls[0])
            tool_info["tool_name"] = first_call.get("name", "unknown")
        except:
            # Fallback to simple pattern matching
            if "zoom" in response.lower():
                tool_info["tool_name"] = "zoom"
            elif "wait" in response.lower():
                tool_info["tool_name"] = "wait"
            elif "inspect" in response.lower():
                tool_info["tool_name"] = "inspect"
    
    # Extract confidence if present
    conf_before_pattern = r'confidence_before:\s*(\d+\.?\d*)'
    conf_after_pattern = r'confidence_after:\s*(\d+\.?\d*)'
    
    conf_before_match = re.search(conf_before_pattern, response)
    if conf_before_match:
        tool_info["confidence_before"] = float(conf_before_match.group(1))
    
    conf_after_match = re.search(conf_after_pattern, response)
    if conf_after_match:
        tool_info["confidence_after"] = float(conf_after_match.group(1))
    
    return tool_info


class ArcVisionRewardScore:
    """Arc Vision RL composite reward model.
    
    Implements the 3-component reward structure:
    1. Task performance (IoU)
    2. Tool effectiveness (confidence gain)
    3. Gating penalty (prevent tool abuse)
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 reward_weights: Dict[str, float] = None,
                 tool_penalties: Dict[str, float] = None):
        """Initialize Arc Vision reward model.
        
        Args:
            confidence_threshold: Threshold for tool invocation
            reward_weights: Weights for reward components
            tool_penalties: Penalties for different failure modes
        """
        self.confidence_threshold = confidence_threshold
        self.weights = reward_weights or {
            "task": 0.6,
            "tool": 0.3,
            "gate": 0.1
        }
        self.penalties = tool_penalties or {
            "unnecessary_tool": -0.5,
            "missed_opportunity": -0.3,
            "ineffective_tool": -0.2,
            "excessive_tools": -0.4
        }
    
    def __call__(self, questions: List[str], responses: List[str], reward_model: Any) -> List[float]:
        """Compute rewards for a batch of responses.
        
        Args:
            questions: List of prompts/questions
            responses: List of model responses
            reward_model: Dictionary containing ground truth and other metadata
            
        Returns:
            List of scalar rewards
        """
        rewards = []
        
        for i, (question, response) in enumerate(zip(questions, responses)):
            # Get ground truth bbox
            if isinstance(reward_model, list):
                gt_data = reward_model[i]
            else:
                gt_data = reward_model
            
            # Extract ground truth bbox
            if "ground_truth" in gt_data:
                gt_bbox = np.array(json.loads(gt_data["ground_truth"]), dtype=np.float32)
            else:
                # Skip if no ground truth
                rewards.append(0.0)
                continue
            
            # Parse model output
            pred_bbox, bbox_success = parse_bbox_from_response(response)
            tool_info = parse_tool_usage(response)
            
            # Compute reward components
            reward = self._compute_composite_reward(
                pred_bbox=pred_bbox,
                gt_bbox=gt_bbox,
                bbox_success=bbox_success,
                tool_info=tool_info
            )
            
            rewards.append(reward)
        
        return rewards
    
    def _compute_composite_reward(self,
                                  pred_bbox: np.ndarray,
                                  gt_bbox: np.ndarray,
                                  bbox_success: bool,
                                  tool_info: Dict[str, Any]) -> float:
        """Compute the 3-component composite reward.
        
        Args:
            pred_bbox: Predicted bounding box
            gt_bbox: Ground truth bounding box
            bbox_success: Whether bbox was successfully parsed
            tool_info: Tool usage information
            
        Returns:
            Composite reward score
        """
        # Component 1: Task performance (IoU-based)
        if bbox_success:
            r_task = compute_iou(pred_bbox, gt_bbox)
        else:
            r_task = 0.0  # Failed to produce valid bbox
        
        # Component 2: Tool effectiveness
        r_tool = 0.0
        if tool_info["tool_used"]:
            # Calculate confidence gain
            conf_gain = tool_info["confidence_after"] - tool_info["confidence_before"]
            
            # Reward based on effectiveness
            if tool_info["confidence_before"] < self.confidence_threshold:
                if tool_info["confidence_after"] >= self.confidence_threshold:
                    # Crossed threshold - maximum reward
                    r_tool = 1.0
                elif conf_gain > 0:
                    # Positive gain but didn't cross threshold
                    r_tool = conf_gain * 2.0  # Scale up small gains
                else:
                    # Tool didn't help
                    r_tool = self.penalties["ineffective_tool"]
            else:
                # Already confident but used tool anyway
                r_tool = self.penalties["unnecessary_tool"]
            
            # Penalty for excessive tool use
            if tool_info["tool_calls"] > 2:
                r_tool += self.penalties["excessive_tools"] * (tool_info["tool_calls"] - 2)
        
        # Component 3: Gating penalty
        r_gate = 0.0
        if tool_info["confidence_before"] > self.confidence_threshold and tool_info["tool_used"]:
            # Unnecessary tool use
            r_gate = self.penalties["unnecessary_tool"]
        elif tool_info["confidence_before"] < self.confidence_threshold and not tool_info["tool_used"]:
            # Missed opportunity to use tools
            # Only penalize if task performance is poor
            if r_task < 0.5:
                r_gate = self.penalties["missed_opportunity"]
        
        # Combine components
        total_reward = (
            self.weights["task"] * r_task +
            self.weights["tool"] * r_tool +
            self.weights["gate"] * r_gate
        )
        
        return float(total_reward)


def compute_score(response: str, ground_truth: str, 
                  confidence_threshold: float = 0.7,
                  reward_weights: Dict[str, float] = None) -> float:
    """Convenience function to compute Arc Vision reward for a single sample.
    
    Args:
        response: Model response
        ground_truth: Ground truth bbox as JSON string
        confidence_threshold: Confidence threshold for tool use
        reward_weights: Optional custom reward weights
        
    Returns:
        Reward score
    """
    reward_model = ArcVisionRewardScore(
        confidence_threshold=confidence_threshold,
        reward_weights=reward_weights
    )
    
    rewards = reward_model(
        questions=[""],  # Question not used in computation
        responses=[response],
        reward_model=[{"ground_truth": ground_truth}]
    )
    
    return rewards[0]