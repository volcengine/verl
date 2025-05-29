"""
Unit tests for UISubtaskRewardScorer
"""

from unittest.mock import patch

import numpy as np
import pytest

from orby.reward.subtask import UISubtaskRewardScorer, compute_score


class TestUISubtaskRewardScorer:
    """Test suite for UISubtaskRewardScorer class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.scorer = UISubtaskRewardScorer()

    def test_init(self):
        """Test that the scorer initializes with correct weights and tags."""
        assert self.scorer.reward_model_tags == [
            "reasoning",
            "should_end",
            "goal_achieved",
            "answer",
        ]
        assert self.scorer.executor_tags == ["thinking", "action"]

    def test_check_text_similarity_identical(self):
        """Test text similarity with identical strings."""
        assert self.scorer._check_text_similarity("hello world", "hello world") is True

    def test_check_text_similarity_case_insensitive(self):
        """Test text similarity is case insensitive."""
        assert self.scorer._check_text_similarity("Hello World", "hello world") is True

    def test_check_text_similarity_high_similarity(self):
        """Test text similarity with high similarity strings."""
        assert self.scorer._check_text_similarity("hello world", "hello world!") is True

    def test_check_text_similarity_low_similarity(self):
        """Test text similarity with low similarity strings."""
        assert (
            self.scorer._check_text_similarity("hello world", "completely different")
            is False
        )

    def test_check_text_similarity_custom_threshold(self):
        """Test text similarity with custom threshold."""
        assert (
            self.scorer._check_text_similarity("hello", "helo", threshold=0.6) is True
        )
        assert (
            self.scorer._check_text_similarity("hello", "xyz", threshold=0.6) is False
        )

    def test_score_reward_model_perfect_match(self):
        """Test reward model scoring with perfect match."""
        prediction = "<reasoning>This is the reasoning</reasoning><should_end>true</should_end><goal_achieved>true</goal_achieved><answer>The answer</answer>"
        ground_truth = {
            "reasoning": "This is the reasoning",
            "should_end": "true",
            "goal_achieved": "true",
            "answer": "The answer",
        }

        result = self.scorer._score_reward_model(prediction, ground_truth)

        assert np.isclose(result["score"], 1.0)
        assert np.isclose(result["format"], 1.0)
        assert result["reasoning"] == 1
        assert result["should_end"] == 1
        assert result["goal_achieved"] == 1
        assert result["answer"] == 1

    def test_score_reward_model_should_end_false(self):
        """Test reward model scoring when should_end is false."""
        prediction = "<reasoning>Still working on it</reasoning><should_end>false</should_end><goal_achieved>false</goal_achieved><answer>No answer yet</answer>"
        ground_truth = {
            "reasoning": "Still working on it",
            "should_end": "false",
            "goal_achieved": "false",
            "answer": "Should be ignored",
        }

        result = self.scorer._score_reward_model(prediction, ground_truth)

        # Answer should be 0 because predicted answer ("No answer yet") doesn't match empty string
        # (gt answer is set to empty when should_end is false)
        assert result["answer"] == 0

    def test_score_reward_model_missing_fields(self):
        """Test reward model scoring with missing fields."""
        prediction = "<reasoning>This is the reasoning</reasoning><goal_achieved>true</goal_achieved>"
        ground_truth = {
            "reasoning": "This is the reasoning",
            "should_end": "true",
            "goal_achieved": "true",
            "answer": "The answer",
        }

        result = self.scorer._score_reward_model(prediction, ground_truth)

        # Format score should be 2/4 = 0.5 (2 fields present out of 4)
        assert np.isclose(result["format"], 0.5)

    def test_calculate_coordinates_score_both_none(self):
        """Test coordinates scoring when both are None."""
        score = self.scorer._calculate_coordinates_score(None, None)
        assert np.isclose(score, 1.0)

    def test_calculate_coordinates_score_one_none(self):
        """Test coordinates scoring when one is None."""
        score = self.scorer._calculate_coordinates_score([(10, 20)], None)
        assert np.isclose(score, 0.0)

        score = self.scorer._calculate_coordinates_score(None, [(10, 20)])
        assert np.isclose(score, 0.0)

    def test_calculate_coordinates_score_identical(self):
        """Test coordinates scoring with identical coordinates."""
        coords = [(10, 20), (30, 40)]
        score = self.scorer._calculate_coordinates_score(coords, coords)
        assert np.isclose(score, 1.0)

    def test_calculate_coordinates_score_close(self):
        """Test coordinates scoring with close coordinates."""
        pred_coords = [(10, 20)]
        gt_coords = [(11, 21)]
        score = self.scorer._calculate_coordinates_score(pred_coords, gt_coords)

        # Should be high but not 1.0 due to Gaussian similarity (around 0.78)
        assert 0.7 < score < 1.0

    def test_calculate_coordinates_score_far(self):
        """Test coordinates scoring with far coordinates."""
        pred_coords = [(10, 20)]
        gt_coords = [(100, 200)]
        score = self.scorer._calculate_coordinates_score(pred_coords, gt_coords)

        # Should be very low
        assert score < 0.1

    def test_calculate_action_args_score_both_none(self):
        """Test action args scoring when both are None."""
        score = self.scorer._calculate_action_args_score(None, None)
        assert np.isclose(score, 1.0)

    def test_calculate_action_args_score_one_none(self):
        """Test action args scoring when one is None."""
        score = self.scorer._calculate_action_args_score({"key": "value"}, None)
        assert np.isclose(score, 0.0)

        score = self.scorer._calculate_action_args_score(None, {"key": "value"})
        assert np.isclose(score, 0.0)

    def test_calculate_action_args_score_key_mismatch(self):
        """Test action args scoring with key mismatch."""
        pred_args = {"key1": "value1"}
        gt_args = {"key2": "value2"}
        score = self.scorer._calculate_action_args_score(pred_args, gt_args)
        assert np.isclose(score, 0.0)

    def test_calculate_action_args_score_identical(self):
        """Test action args scoring with identical args."""
        args = {"button": "left", "double": "false"}
        score = self.scorer._calculate_action_args_score(args, args)
        assert np.isclose(score, 1.0)

    def test_calculate_action_args_score_similar(self):
        """Test action args scoring with similar args."""
        pred_args = {"text": "hello world"}
        gt_args = {"text": "hello world!"}
        score = self.scorer._calculate_action_args_score(pred_args, gt_args)

        # Should be high due to text similarity
        assert score > 0.8

    def test_score_executor_perfect_match(self):
        """Test executor scoring with perfect match."""
        prediction = "<thinking>I need to click the button</thinking><action>click(100, 200, button='left', double=False)</action>"
        ground_truth = {
            "thinking": "I need to click the button",
            "action": "click(100, 200, button='left', double=False)",
        }

        result = self.scorer._score_executor(prediction, ground_truth)

        assert np.isclose(result["score"], 1.0)
        assert np.isclose(result["format"], 1.0)
        assert result["thinking"] == 1
        assert result["action_type"] == 1
        assert np.isclose(result["coordinates"], 1.0)
        assert np.isclose(result["action_args"], 1.0)

    def test_score_executor_invalid_action(self):
        """Test executor scoring with invalid action."""
        prediction = "<thinking>I need to do something</thinking><action>invalid_action()</action>"
        ground_truth = {
            "thinking": "I need to do something",
            "action": "click(100, 200)",
        }

        with patch("builtins.print"):  # Mock print to avoid output during tests
            result = self.scorer._score_executor(prediction, ground_truth)

        # Should only get format and thinking scores
        assert result["score"] > 0
        assert result["format"] > 0
        assert result["thinking"] > 0
        assert result["action_type"] == 0
        assert result["coordinates"] == 0
        assert result["action_args"] == 0

    def test_score_reward_model_type(self):
        """Test score method with reward model type ground truth."""
        ground_truth = {
            "reasoning": "Test reasoning",
            "should_end": "true",
            "goal_achieved": "true",
            "answer": "Test answer",
        }

        prediction = "<reasoning>Test reasoning</reasoning><should_end>true</should_end><goal_achieved>true</goal_achieved><answer>Test answer</answer>"

        result = self.scorer.score(prediction, ground_truth)
        assert np.isclose(result["score"], 1.0)
        assert np.isclose(result["format"], 1.0)
        assert result["reasoning"] == 1
        assert result["should_end"] == 1
        assert result["goal_achieved"] == 1
        assert result["answer"] == 1

    def test_score_executor_type(self):
        """Test score method with executor type ground truth."""
        ground_truth = {"thinking": "Test thinking", "action": "click(100, 200)"}

        prediction = (
            "<thinking>Test thinking</thinking><action>click(100, 200)</action>"
        )

        result = self.scorer.score(prediction, ground_truth)
        assert np.isclose(result["score"], 1.0)
        assert np.isclose(result["format"], 1.0)
        assert result["thinking"] == 1
        assert result["action_type"] == 1
        assert np.isclose(result["coordinates"], 1.0)
        assert np.isclose(result["action_args"], 1.0)

    def test_score_invalid_ground_truth(self):
        """Test score method with invalid ground truth."""
        ground_truth = {"invalid_key": "invalid_value"}

        prediction = "Some prediction"

        with pytest.raises(ValueError, match="Invalid ground truth type"):
            self.scorer.score(prediction, ground_truth)

    def test_compute_score_function(self):
        """Test the standalone compute_score function."""
        prediction = "<reasoning>Test</reasoning><should_end>true</should_end><goal_achieved>true</goal_achieved><answer>Test</answer>"
        ground_truth = {
            "reasoning": "Test",
            "should_end": "true",
            "goal_achieved": "true",
            "answer": "Test",
        }

        result = compute_score(prediction, ground_truth)

        assert "score" in result
        assert 0 <= result["score"] <= 1

    def test_score_reward_model_with_none_values(self):
        """Test reward model scoring when extract returns None values."""
        prediction = "No valid tags"
        ground_truth = {
            "reasoning": "Expected reasoning",
            "should_end": "true",
            "goal_achieved": "true",
            "answer": "Expected answer",
        }

        result = self.scorer._score_reward_model(prediction, ground_truth)

        # All scores should be 0
        assert np.isclose(result["format"], 0.0)
        assert result["reasoning"] == 0
        assert result["should_end"] == 0
        assert result["goal_achieved"] == 0
        assert result["answer"] == 0
