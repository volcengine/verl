"""
Reward scoring for UI Subtask task
"""

from difflib import SequenceMatcher

import numpy as np

from orby.utils.action_utils import get_action_info, extract_content_by_tags


class UISubtaskRewardScorer:
    """Reward scorer for UI Subtask task."""

    def __init__(self):
        super().__init__()
        self.reward_model_weights = {
            "format": 0.1,
            "reasoning": 0.1,
            "should_end": 0.3,
            "goal_achieved": 0.3,
            "answer": 0.2,
        }
        self.executor_weights = {
            "format": 0.1,
            "thinking": 0.1,
            "action_type": 0.1,
            "coordinates": 0.5,
            "action_args": 0.2,
        }
        self.reward_model_tags = ["reasoning", "should_end", "goal_achieved", "answer"]
        self.executor_tags = ["thinking", "action"]

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
        similarity = SequenceMatcher(
            None, pred_content.lower(), gt_content.lower()
        ).ratio()
        return similarity >= threshold

    def _score_reward_model(self, prediction: str, ground_truth: dict) -> dict:
        """
        Score the prediction against ground truth for reward model.

        Args:
            prediction: Model prediction string
            ground_truth: Dictionary containing ground truth information.
                - reasoning: str
                - should_end: Literal["true", "false"]
                - goal_achieved: Literal["true", "false"]
                - answer: str

        Returns:
            Dictionary containing:
                - score: Overall score (0-1)
                - other individual check scores
        """
        pred_dict = extract_content_by_tags(prediction, self.reward_model_tags)

        # check format
        format_score = sum(
            1 for value in pred_dict.values() if value is not None
        ) / len(self.reward_model_tags)

        # Convert everything to string, even None
        pred_dict = {key: str(value) for key, value in pred_dict.items()}
        gt_dict = {key: str(value) for key, value in ground_truth.items()}

        pred_dict["should_end"] = pred_dict["should_end"].lower() == "true"
        pred_dict["goal_achieved"] = pred_dict["goal_achieved"].lower() == "true"
        gt_dict["should_end"] = gt_dict["should_end"].lower() == "true"
        gt_dict["goal_achieved"] = gt_dict["goal_achieved"].lower() == "true"

        # Remove gt answer if should_end is false
        if not gt_dict["should_end"]:
            gt_dict["answer"] = ""

        try:
            reasoning_score = int(
                self._check_text_similarity(
                    pred_dict["reasoning"], gt_dict["reasoning"]
                )
            )
        except Exception as e:
            print(f"Error calculating reasoning score: {e}")
            reasoning_score = 0
        try:
            should_end_score = int(pred_dict["should_end"] == gt_dict["should_end"])
        except Exception as e:
            print(f"Error calculating should end score: {e}")
            should_end_score = 0
        try:
            goal_achieved_score = int(
                pred_dict["goal_achieved"] == gt_dict["goal_achieved"]
            )
        except Exception as e:
            print(f"Error calculating goal achieved score: {e}")
            goal_achieved_score = 0
        try:
            answer_score = int(
                self._check_text_similarity(pred_dict["answer"], gt_dict["answer"])
            )
        except Exception as e:
            print(f"Error calculating answer score: {e}")
            answer_score = 0

        score = (
            format_score * self.reward_model_weights["format"]
            + reasoning_score * self.reward_model_weights["reasoning"]
            + should_end_score * self.reward_model_weights["should_end"]
            + goal_achieved_score * self.reward_model_weights["goal_achieved"]
            + answer_score * self.reward_model_weights["answer"]
        )
        details = {
            "score": score,
            "format": format_score,
            "reasoning": reasoning_score,
            "should_end": should_end_score,
            "goal_achieved": goal_achieved_score,
            "answer": answer_score,
        }

        return details

    def _calculate_coordinates_score(
        self,
        pred_coordinates: list[tuple[float, float]] | None,
        gt_coordinates: list[tuple[float, float]] | None,
    ) -> float:
        """
        Calculate the score for the coordinates of the action.
        We use a Gaussian similarity score with a sigma of 2 for all coordinates.

        Args:
            pred_coordinates: Predicted coordinates
            gt_coordinates: Ground truth coordinates

        Returns:
            Score for the coordinates between 0 and 1
        """
        # If coordinates are not necessary and the model predicted None
        # we should not penalize the model
        if pred_coordinates is None and gt_coordinates is None:
            return 1.0
        # If the model predicted None but the ground truth is not None, or vice versa,
        # we should penalize the model
        if pred_coordinates is None or gt_coordinates is None:
            return 0.0

        # If the length of the coordinates is different, we should penalize the model
        if len(pred_coordinates) != len(gt_coordinates):
            return 0.0

        scores = []
        for pred_coord, gt_coord in zip(pred_coordinates, gt_coordinates):
            # Use a Gaussian similarity score with a sigma of 2
            # TODO: if we can get the bounding box information, investigate which reward is better
            sigma = 2
            pred = np.asarray(pred_coord)
            truth = np.asarray(gt_coord)
            d2 = np.sum((pred - truth) ** 2)
            score = np.exp(-d2 / (2 * sigma**2))
            scores.append(score)

        return np.mean(scores)

    def _calculate_action_args_score(
        self, pred_args: dict[str, str] | None, gt_args: dict[str, str] | None
    ) -> float:
        """
        Calculate the score for the action arguments.
        We treat every argument as a string and use the textual similarity score.

        Args:
            pred_args: Predicted arguments
            gt_args: Ground truth arguments

        Returns:
            Score for the arguments between 0 and 1
        """
        # If both are None, we should not penalize the model
        if pred_args is None and gt_args is None:
            return 1.0
        # If one is None but the other is not, we should penalize the model
        if pred_args is None or gt_args is None:
            return 0.0
        # if there is a key mismatch, we should penalize the model
        if pred_args.keys() != gt_args.keys():
            return 0.0

        scores = []
        for key in pred_args.keys():
            score = self._check_text_similarity(pred_args[key], gt_args[key])
            scores.append(score)

        return np.mean(scores)

    def _score_executor(self, prediction: str, ground_truth: dict) -> dict:
        """
        Score the prediction against ground truth for executor.

        Args:
            prediction: Model prediction string
            ground_truth: Dictionary containing ground truth information.
                - thinking: str
                - action: str

        Returns:
            Dictionary containing:
                - score: Overall score (0-1)
                - other individual check scores
        """
        pred_dict = extract_content_by_tags(prediction, self.executor_tags)
        format_score = sum(
            1 for value in pred_dict.values() if value is not None
        ) / len(self.executor_tags)

        # Convert everything to string, even None
        pred_dict = {key: str(value) for key, value in pred_dict.items()}
        gt_dict = {key: str(value) for key, value in ground_truth.items()}

        thinking_score = int(
            self._check_text_similarity(pred_dict["thinking"], gt_dict["thinking"])
        )

        parser_error = False
        try:
            pred_action_info = get_action_info(pred_dict["action"])
            gt_action_info = get_action_info(gt_dict["action"])
            action_type_score = int(
                pred_action_info["action_type"] == gt_action_info["action_type"]
            )
        except Exception as e:
            # If the action is not in the action space, we should penalize the model and exit early
            print(f"Error with action type parsing: {e}")
            parser_error = True
            action_type_score = 0

        coordinates_score = 0
        action_args_score = 0
        if not parser_error:
            try:
                coordinates_score = self._calculate_coordinates_score(
                    pred_action_info["coordinates"], gt_action_info["coordinates"]
                )
            except Exception as e:
                print(f"Error calculating coordinates score: {e}")
            try:
                action_args_score = self._calculate_action_args_score(
                    pred_action_info["args"], gt_action_info["args"]
                )
            except Exception as e:
                print(f"Error calculating action args score: {e}")

        score = (
            format_score * self.executor_weights["format"]
            + thinking_score * self.executor_weights["thinking"]
            + action_type_score * self.executor_weights["action_type"]
            + coordinates_score * self.executor_weights["coordinates"]
            + action_args_score * self.executor_weights["action_args"]
        )
        details = {
            "score": score,
            "format": format_score,
            "thinking": thinking_score,
            "action_type": action_type_score,
            "coordinates": coordinates_score,
            "action_args": action_args_score,
        }

        return details

    def score(self, prediction: str, ground_truth: dict) -> dict:
        """Score the prediction against ground truth.

        Args:
            prediction: Model prediction string
            ground_truth: Dictionary containing ground truth information.
            There can be 2 types of ground truth:
            - reward_model: with fields
                - reasoning: str
                - should_end: Literal["true", "false"]
                - goal_achieved: Literal["true", "false"]
                - answer: str
            - executor: with fields
                - thinking: str
                - action: str

        Returns:
            Dictionary containing:
                - score: Overall score (0-1)
                - other individual check scores
        """
        gt_keys = ground_truth.keys()
        if (
            "reasoning" in gt_keys
            and "should_end" in gt_keys
            and "goal_achieved" in gt_keys
            and "answer" in gt_keys
        ):
            result = self._score_reward_model(prediction, ground_truth)
        elif "thinking" in gt_keys and "action" in gt_keys:
            result = self._score_executor(prediction, ground_truth)
        else:
            raise ValueError("Invalid ground truth type")

        return result


def compute_score(prediction: str, ground_truth: dict) -> dict:
    """Compute score for a single prediction.

    Args:
        prediction: Prediction string
        ground_truth: Dictionary containing ground truth information
    """
    scorer = UISubtaskRewardScorer()
    result = scorer.score(prediction, ground_truth)
    return result


def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "subtask_direct_distill":
        from orby.reward import subtask

        return subtask.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError
