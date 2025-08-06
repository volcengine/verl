# tests/workers/reward_model/test_reward_logic_on_cpu.py

import torch

# Correctly import the functions from their new location in the core library
from verl.utils.reward_process import reward_preprocess, reward_postprocess

# ==============================================================================
#  Tests for `reward_preprocess`
# ==============================================================================

class TestRewardPreprocess:
    """Groups tests for the reward_preprocess function."""

    def test_basic_functionality_and_mocked_score(self, mocker):
        """
        Tests if reward_preprocess correctly constructs chat prompts and calls
        the scoring function.
        """
        # 1. Arrange: Mock the external scoring function to isolate our test.
        # We patch the function *where it is looked up*, not where it is defined.
        mock_compute_score = mocker.patch('verl.utils.reward_process.compute_score')
        # Define a predictable return value for the mocked function.
        mock_compute_score.return_value = 0.8

        questions = ["What is the capital of France?"]
        answers = ["Paris is the city of light."]
        ground_truths = ["Paris"]

        # 2. Act: Call the function under test.
        chats, extra_scores = reward_preprocess(questions, answers, ground_truths)

        # 3. Assert: Verify the outputs and interactions.
        # Verify that the scoring function was called correctly.
        mock_compute_score.assert_called_once_with(answers[0], ground_truths[0])

        # Check types and shapes
        assert isinstance(chats, list)
        assert len(chats) == 1
        assert isinstance(extra_scores, torch.Tensor)
        assert extra_scores.shape == (1,)
        
        # Check that the tensor contains the value from our mocked function.
        torch.testing.assert_close(extra_scores[0], torch.tensor(0.8))

        # Check chat structure and content
        chat_sample = chats[0]
        assert len(chat_sample) == 2
        assert chat_sample[0]['role'] == 'system'
        assert chat_sample[1]['role'] == 'user'
        user_content = chat_sample[1]['content']
        assert answers[0] in user_content
        assert ground_truths[0] in user_content
        # The real implementation comments out the question, so we assert it's NOT there.
        assert questions[0] not in user_content

    def test_batch_processing_with_mocked_scores(self, mocker):
        """
        Tests if reward_preprocess handles a batch of inputs correctly, using
        mocked scores.
        """
        # 1. Arrange: Mock the scoring function to return different values on
        # consecutive calls.
        mock_compute_score = mocker.patch('verl.utils.reward_process.compute_score')
        mock_compute_score.side_effect = [0.9, 0.1] # First call returns 0.9, second returns 0.1

        questions = ["Q1", "Q2"]
        answers = ["A1", "A2"]
        ground_truths = ["G1", "G2"]

        # 2. Act
        chats, extra_scores = reward_preprocess(questions, answers, ground_truths)

        # 3. Assert
        assert extra_scores is not None
        assert len(chats) == 2
        assert extra_scores.shape == (2,)
        
        # Check that the scores in the tensor match the side_effect list.
        expected_scores = torch.tensor([0.9, 0.1], dtype=torch.float32)
        torch.testing.assert_close(extra_scores, expected_scores)

        # Verify the mock was called twice with the correct arguments.
        assert mock_compute_score.call_count == 2
        mock_compute_score.assert_any_call("A1", "G1")
        mock_compute_score.assert_any_call("A2", "G2")

    def test_empty_input(self):
        """
        Tests if reward_preprocess handles empty input lists correctly.
        """
        chats, extra_scores = reward_preprocess([], [], [])
        assert extra_scores is not None
        assert len(chats) == 0
        assert extra_scores.shape == (0,)

    def test_mismatched_lengths(self):
        """
        Tests if reward_preprocess raises an error when input lists have different lengths.
        This behavior comes from the zip() function.
        """
        # This will not raise an error, zip will just stop at the shortest list.
        # Let's test the actual behavior.
        questions = ["Q1", "Q2"]
        answers = ["A1"]
        ground_truths = ["G1", "G2"]
        chats, _ = reward_preprocess(questions, answers, ground_truths)
        # The loop runs only once because 'answers' has only one element.
        assert len(chats) == 1

# ==============================================================================
#  Tests for `reward_postprocess`
# ==============================================================================

class TestRewardPostprocess:
    """Groups tests for the reward_postprocess function."""

    def test_parsing_and_blending_with_scores_tensor(self):
        """
        Tests if it correctly parses scores from text and blends them with scores_tensor.
        """
        gen_rm_responses = [
            "The final answer is \\boxed{4}",       # Parsable score
            "I am not entirely sure about this one." # Unparsable, defaults to 0
        ]
        scores_tensor = torch.tensor([1.0, 0.2], dtype=torch.float32)

        final_scores = reward_postprocess(gen_rm_responses, scores_tensor)

        assert final_scores.shape == (2,)
        # Expected for sample 1: (parsed_rm_score=4.0 + pre_score=1.0 * 5) / 2 = 4.5
        torch.testing.assert_close(final_scores[0], torch.tensor(4.5))
        # Expected for sample 2: (parsed_rm_score=0.0 + pre_score=0.2 * 5) / 2 = 0.5
        torch.testing.assert_close(final_scores[1], torch.tensor(0.5))

    def test_operation_without_scores_tensor(self):
        """
        Tests if the function works correctly when scores_tensor is None.
        """
        gen_rm_responses = ["The score is \\boxed{5}", "I'd rate it a \\boxed{2}"]
        final_scores = reward_postprocess(gen_rm_responses)
        expected_scores = torch.tensor([5.0, 2.0])
        torch.testing.assert_close(final_scores, expected_scores)

    def test_edge_cases(self):
        """
        Tests edge cases like score clamping, malformed text, and multiple boxes.
        """
        gen_rm_responses = [
            "Amazing! \\boxed{10}",                 # Score > 5, should be clamped to 5.0
            "Here is the result: \\boxed{bad}",     # Malformed, regex `\d+` won't match, defaults to 0.0
            "Final result: \\boxed{1}, thought: \\boxed{0}", # Should take the last one: 0.0
            "A negative score \\boxed{-5}",         # Malformed, regex `\d+` won't match '-', defaults to 0.0
            "This is the only one \\boxed{3}"       # A normal case
        ]
        
        final_scores = reward_postprocess(gen_rm_responses)

        expected_scores = torch.tensor([
            5.0,  # 10 is clamped to 5.0
            0.0,  # 'bad' is not parsable, defaults to 0.0
            0.0,  # The last parsable number is 0
            0.0,  # '-' is not a digit, so no match, defaults to 0.0
            3.0
        ])
        torch.testing.assert_close(final_scores, expected_scores)

    def test_empty_responses(self):
        """
        Tests if reward_postprocess handles empty response lists correctly.
        """
        final_scores = reward_postprocess([])
        assert final_scores.shape == (0,)
