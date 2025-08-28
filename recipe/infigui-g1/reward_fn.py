# Copyright 2025 Individual Contributor: InfiX.ai
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

import json
import logging
import math
import re
from itertools import combinations

FMT_RATIO = 1.0
ACC_RATIO = 1.0


# ============================================================================
# Utility Functions
# ============================================================================


def extract_think_format(predict_str: str) -> None | dict[str, str]:
    """
    Check if the predicted string meets format requirements and extract thinking and answer parts.

    Args:
        predict_str: The predicted string

    Returns:
        If format requirements are met, returns a dictionary containing thinking and answer parts;
        otherwise returns None
    """
    if not predict_str or not isinstance(predict_str, str):
        return None

    # Check if <think> is at the beginning
    if not predict_str.startswith("<think>"):
        return None

    # Check if there is <think>...</think> format
    pattern = r"<think>(.*?)</think>"
    think_match = re.search(pattern, predict_str, re.DOTALL)
    if not think_match:
        return None

    if predict_str.count("<think>") != 1 or predict_str.count("</think>") != 1:
        return None

    # Extract thinking content
    think_content = think_match.group(1).strip()
    if not think_content:
        return None

    # Get content after </think>
    think_end_pos = predict_str.find("</think>") + len("</think>")
    post_think_content = predict_str[think_end_pos:].strip()

    # Check if there is non-empty content after </think>
    if not post_think_content:
        return None

    return {"think": think_content, "answer": post_think_content}


def extract_and_parse_json(input_string, wrapper):
    """
    Try to extract and parse JSON from a string.

    Args:
        input_string: The input string
        wrapper: JSON wrapper symbols, can be '{}' or '[]'

    Returns:
        Parsed JSON object, returns None if parsing fails
    """
    if len(wrapper) != 2:
        raise ValueError("Wrapper must be exactly two characters long")

    start_char, end_char = wrapper
    start_index = input_string.find(start_char)

    if start_index == -1:
        return None

    # Find the matching end character by balancing brackets/braces
    balance = 1
    end_index = -1
    for i in range(start_index + 1, len(input_string)):
        if input_string[i] == start_char:
            balance += 1
        elif input_string[i] == end_char:
            balance -= 1

        if balance == 0:
            end_index = i
            break

    if end_index == -1:
        return None

    json_string = input_string[start_index : end_index + 1]

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None


# ============================================================================
# AER Reward Functions
# ============================================================================


def _extract_verifiable_answer(answer):
    """
    Extract and verify the format of the point list from the answer string.

    A valid format is a JSON list of dictionaries, where each dictionary
    has a "point_2d" key with a list of two numbers as the value.

    Args:
        answer: The answer string to extract points from

    Returns:
        List of valid points or None if format is invalid
    """
    points = extract_and_parse_json(answer, "[]")
    if points is None or not isinstance(points, list):
        return None

    # Verify each point in the list
    for point in points:
        if isinstance(point, dict) and "point_2d" in point:
            point_2d = point["point_2d"]
            if isinstance(point_2d, list) and len(point_2d) == 2:
                continue

        # If any point is malformed, the whole answer is invalid
        return None

    return points


def _format_reward(answer):
    """
    Calculate the format reward for 'point' type data.

    This function is now primarily used as a check to see if the format is valid.

    Args:
        answer: The answer string to validate

    Returns:
        Tuple of (reward, is_collinear) where reward is 1.0 for valid format, 0.0 otherwise
    """
    points = _extract_verifiable_answer(answer)
    if points is None:
        return 0.0, 0

    points_2d = [item["point_2d"] for item in points]
    if _check_collinear(points_2d):
        return 0.0, 1

    return 1.0, 0


def _check_collinear(points_2d):
    """
    Check if 3 or more points in the list are collinear on any straight line.

    This uses the cross-product method to avoid division and handle all line types.

    Args:
        points_2d: A list of [x, y] coordinates

    Returns:
        True if 3 or more points are collinear, False otherwise
    """
    if len(points_2d) < 3:
        return False

    # Iterate through all unique combinations of 3 points
    for p1, p2, p3 in combinations(points_2d, 3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Check for collinearity using the cross-product method.
        # If (y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1), the points are collinear.
        # This is equivalent to checking if the area of the triangle formed by the points is 0.
        if math.isclose((y2 - y1) * (x3 - x1), (y3 - y1) * (x2 - x1)):
            return True

    return False


def _accuracy_reward(answer, ground_truth):
    """
    Calculate the accuracy reward based on the symmetric zero-centered formula.

    The reward is in the range [-1, 1].

    Args:
        answer: The answer string containing predicted points
        ground_truth: Ground truth bounding box dictionary

    Returns:
        Tuple containing:
        - accuracy (float): The calculated reward
        - extracted_answer (str): The JSON string of the predicted points
        - num_pred (int): The number of predicted points
        - first_correct (int): 1 if the first predicted point is correct, 0 otherwise
    """
    pred_points = _extract_verifiable_answer(answer)

    # If no valid points are extracted, this is considered a format error, return -1 reward
    if pred_points is None:
        return -1.0, "", 0, 0

    num_pred = len(pred_points)
    extracted_answer = json.dumps(pred_points)

    if num_pred == 0:
        return -1.0, extracted_answer, 0, 0

    # Find the rank 'k' of the first correct point
    first_correct_rank = -1
    for i, item in enumerate(pred_points):
        point_2d = item["point_2d"]
        if (
            ground_truth["x1"] <= point_2d[0] <= ground_truth["x2"]
            and ground_truth["y1"] <= point_2d[1] <= ground_truth["y2"]
        ):
            first_correct_rank = i + 1  # 1-based index
            break

    # Calculate reward based on the zero-centered symmetric formula
    accuracy = 0.0
    if first_correct_rank != -1:
        # Case a: Correct point found (Positive reward space)
        k = first_correct_rank
        accuracy = 1.0 / math.sqrt(num_pred * k)
    else:
        # Case b: No correct point found (Negative reward space)
        accuracy = -1.0 / num_pred

    first_correct = 1 if first_correct_rank == 1 else 0

    return accuracy, extracted_answer, num_pred, first_correct


def calculate_point_reward(solution_str, ground_truth, extra_info=None, fmt_ratio=1.0, acc_ratio=1.0, **kwargs):
    """
    Calculate the final reward for 'point' type data.

    Implements the full logic including format checks, collinearity checks,
    and the zero-centered symmetric reward calculation.

    Args:
        solution_str: The solution string from the model
        ground_truth: Ground truth data
        extra_info: Extra information dictionary
        fmt_ratio: Format reward ratio
        acc_ratio: Accuracy reward ratio
        **kwargs: Additional keyword arguments

    Returns:
        Dictionary containing detailed reward information
    """
    if extra_info.get("no_think", False):
        answer = solution_str
    else:
        solution_dict = extract_think_format(solution_str)
        # If the overall 'think'/'answer' format is wrong, return score of -1
        if solution_dict is None:
            return {
                "score": -1.0,
                "format": 0.0,
                "accuracy": -1.0,
                "pred": "",
                "num_pred": 0,
                "has_correct": 0,
                "first_correct": 0,
                "only_correct": 0,
                "is_collinear": 0,
            }

        answer = solution_dict["answer"]

    # Reuse _format_reward to check the format of the 'answer' part
    # If it's invalid, return score of -1
    format_reward, is_collinear = _format_reward(answer)
    if format_reward == 0.0:
        return {
            "score": -1.0,
            "format": 0.0,
            "accuracy": -1.0,
            "pred": "",
            "num_pred": 0,
            "has_correct": 0,
            "first_correct": 0,
            "only_correct": 0,
            "is_collinear": is_collinear,
        }

    # If format is OK, calculate the accuracy reward
    accuracy_reward, extracted_answer, num_pred, first_correct = _accuracy_reward(answer, ground_truth)

    return {
        "score": fmt_ratio * format_reward + acc_ratio * accuracy_reward,
        "format": format_reward,
        "accuracy": accuracy_reward,
        "pred": extracted_answer,
        "num_pred": num_pred,
        "has_correct": 1 if accuracy_reward > 0 else 0,
        "first_correct": first_correct,
        "only_correct": 1 if num_pred == 1 and accuracy_reward > 0 else 0,
        "is_collinear": 0,
    }


# ============================================================================
# AER Reward Handler Registry
# ============================================================================

# Dictionary to map data_source to the respective reward calculation function
AER_REWARD_HANDLERS = {
    "point": calculate_point_reward,
}


def aer_gui_reward_function(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Main reward function dispatcher for the Adaptive Exploration Reward (AER) system.

    Delegates reward calculation to specific functions based on the data_source using a dictionary lookup.

    Args:
        data_source: The source or type of the data (e.g., "point", "bbox")
        solution_str: The solution string generated by the model
        ground_truth: The ground truth data
        extra_info: Any extra information passed along (optional)
        **kwargs: Additional keyword arguments that might be passed from the PPO trainer config

    Returns:
        Dictionary containing detailed reward information with keys:
        - score: The final calculated reward score
        - format: Format validation score
        - accuracy: Accuracy score
        - pred: Extracted prediction string
        - num_pred: Number of predictions
        - has_correct: Whether any correct prediction exists
        - first_correct: Whether first prediction is correct
        - only_correct: Whether only one correct prediction exists
        - is_collinear: Whether points are collinear (for point type)
    """
    handler = AER_REWARD_HANDLERS.get(data_source, None)

    if handler:
        try:
            return handler(
                solution_str, ground_truth, extra_info=extra_info, fmt_ratio=FMT_RATIO, acc_ratio=ACC_RATIO, **kwargs
            )
        except Exception as e:
            logging.exception(
                f"Error executing reward handler for data_source '{data_source}': {e}",
            )
            return {
                "score": -1.0,
                "format": 0.0,
                "accuracy": -1.0,
                "pred": "",
                "num_pred": 0,
                "has_correct": 0,
                "first_correct": 0,
                "only_correct": 0,
                "is_collinear": 0,
            }  # Return a default penalty score on error
    else:
        raise ValueError(f"Unknown data_source: '{data_source}'. No specific reward handler defined.")
