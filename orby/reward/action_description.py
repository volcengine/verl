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
Reward scoring for Action Description task
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional


class ActionDescriptionRewardScorer:
    """Reward scorer for Action Description task."""

    def __init__(self):
        super().__init__()
        self.thinking_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self.answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        # Updated pattern to handle both 2-arg and 3-arg formats with decimal numbers
        self.action_pattern = re.compile(
            r"(\w+)\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)(?:,\s*'[^']*')?\)"
        )
        self.keyboard_pattern = re.compile(r"keyboard_type\('([^']*)'\)")

    def _extract_action_info(
        self, action_str: str
    ) -> Tuple[str, Optional[float], Optional[float], Optional[str]]:
        """Extract action type and parameters from action string.

        Args:
            action_str: Action string in format "action_type(x, y)" or "keyboard_type('content')"
                or "select_option(x, y, 'string')"

        Returns:
            Tuple of (action_type, x, y, content)
        """
        # Try keyboard_type pattern first
        keyboard_match = self.keyboard_pattern.match(action_str.strip())
        if keyboard_match:
            return "keyboard_type", None, None, keyboard_match.group(1)

        # Try coordinate-based action pattern
        match = self.action_pattern.match(action_str.strip())
        if not match:
            return "", None, None, None
        action_type, x, y = match.groups()
        return action_type, float(x), float(y), None

    def _check_coordinates_in_bbox(
        self, x: int, y: int, bbox: List[int], tolerance: int = 0
    ) -> bool:
        """Check if coordinates are within bounding box with tolerance.

        Args:
            x: X coordinate
            y: Y coordinate
            bbox: Bounding box [x1, y1, x2, y2]
            tolerance: Pixel tolerance for coordinate matching

        Returns:
            True if coordinates are within bbox with tolerance
        """
        x1, y1, x2, y2 = bbox
        return (
            x1 - tolerance <= x <= x2 + tolerance
            and y1 - tolerance <= y <= y2 + tolerance
        )

    def _check_scroll_values(
        self, pred_x: int, pred_y: int, gt_x: int, gt_y: int
    ) -> float:
        """Check if scroll values match in sign.

        Args:
            pred_x: Predicted x scroll value
            pred_y: Predicted y scroll value
            gt_x: Ground truth x scroll value
            gt_y: Ground truth y scroll value

        Returns:
            float: Score of 1.0 if signs match, 0.0 otherwise
        """
        # Check if signs match for both x and y
        x_sign_match = (pred_x * gt_x > 0) or (pred_x == 0 and gt_x == 0)
        y_sign_match = (pred_y * gt_y > 0) or (pred_y == 0 and gt_y == 0)

        # Return 1.0 if both signs match, 0.0 otherwise
        return 1.0 if (x_sign_match and y_sign_match) else 0.0

    def _check_text_similarity(
        self, pred_content: str, gt_content: str, threshold: float = 0.8
    ) -> bool:
        """Check if predicted content matches ground truth content using text similarity.

        Args:
            pred_content: Predicted content string
            gt_content: Ground truth content string
            threshold: Minimum similarity score to consider as match

        Returns:
            True if similarity score is above threshold
        """
        if pred_content is None or gt_content is None:
            return False
        similarity = SequenceMatcher(
            None, pred_content.lower(), gt_content.lower()
        ).ratio()
        return similarity >= threshold

    def score(self, prediction: str, ground_truth: Dict) -> Dict:
        """Score the prediction against ground truth.

        Args:
            prediction: Model prediction string
            ground_truth: Dictionary containing ground truth information
                - action: Ground truth action string
                - bbox: Optional bounding box [x1, y1, x2, y2]
                - content: Optional content for keyboard_type actions
                - similarity_threshold: Optional threshold for text similarity

        Returns:
            Dictionary containing:
                - score: Overall score (0-1)
                - details: Dictionary with individual check results
        """
        # Check 1: Format validation
        has_thinking = bool(self.thinking_pattern.search(prediction))
        answer_match = self.answer_pattern.search(prediction)
        has_answer = bool(answer_match)

        # Check 2: Action type validation
        gt_action_type, gt_x, gt_y, gt_content = self._extract_action_info(
            ground_truth["action"]
        )
        pred_answer = answer_match.group(1).strip() if answer_match else ""
        pred_action_type, pred_x, pred_y, pred_content = self._extract_action_info(
            pred_answer
        )
        action_type_correct = pred_action_type == gt_action_type

        # Check 3: Content/Coordinate validation
        content_correct = False
        coordinates_correct = False
        coord_score = 0.0

        if gt_action_type == "keyboard_type":
            # For keyboard_type actions, check content similarity
            similarity_threshold = ground_truth.get("similarity_threshold", 0.8)
            content_correct = self._check_text_similarity(
                pred_content, gt_content, similarity_threshold
            )
            coord_score = 1.0 if content_correct else 0.0
        elif gt_action_type == "scroll":
            # For scroll actions, check scroll values
            if (
                pred_x is not None
                and pred_y is not None
                and gt_x is not None
                and gt_y is not None
            ):
                coord_score = self._check_scroll_values(pred_x, pred_y, gt_x, gt_y)
        else:
            # For other coordinate-based actions, check bbox
            bbox = ground_truth.get("bbox")
            if bbox is not None and pred_x is not None and pred_y is not None:
                coordinates_correct = self._check_coordinates_in_bbox(
                    pred_x, pred_y, bbox
                )
                coord_score = 1.0 if coordinates_correct else 0.0

        # Calculate overall score
        format_score = 1.0 if (has_thinking and has_answer) else 0.0
        action_score = 1.0 if action_type_correct else 0.0

        # Weight the scores (can be adjusted based on importance)
        weights = {"format": 0.2, "action_type": 0.3, "coordinates": 0.5}

        overall_score = (
            weights["format"] * format_score
            + weights["action_type"] * action_score
            + weights["coordinates"] * coord_score
        )

        details = {
            "score": overall_score,
            "has_thinking": has_thinking,
            "has_answer": has_answer,
            "format_score": format_score,
            "action_type_predicted": str(pred_action_type),
            "action_type_ground_truth": str(gt_action_type),
            "action_type_score": action_score,
        }

        if gt_action_type == "keyboard_type":
            details.update(
                {
                    "arg_predicted": pred_content,
                    "arg_ground_truth": gt_content,
                    "arg_score": coord_score,
                }
            )
        elif gt_action_type == "scroll":
            details.update(
                {
                    "arg_predicted": str((pred_x, pred_y)),
                    "arg_ground_truth": str((gt_x, gt_y)),
                    "arg_score": coord_score,
                }
            )
        else:
            details.update(
                {
                    "arg_predicted": str((pred_x, pred_y)),
                    "arg_ground_truth": str(ground_truth.get("bbox")),
                    "arg_score": coord_score,
                }
            )

        return details


def compute_score(prediction: str, ground_truth: Dict) -> Dict:
    """Compute score for a single prediction.

    Args:
        prediction: Prediction string
        ground_truth: Dictionary containing ground truth information
            - action: Ground truth action string
            - bbox: Optional bounding box [x1, y1, x2, y2]
            - content: Optional content for keyboard_type actions
            - similarity_threshold: Optional threshold for text similarity

    Returns:
        Dictionary containing:
            - score: Overall score (0-1)
            - details: Dictionary with individual check results
    """
    scorer = ActionDescriptionRewardScorer()
    result = scorer.score(prediction, ground_truth)
    return result


def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    if data_source in ["action_description"]:
        from orby.reward import action_description

        return action_description.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError
