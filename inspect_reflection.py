import sys
import torch
import random
from collections import defaultdict
from transformers import AutoTokenizer

# Import the necessary modules
sys.path.append('~/verl')
from verl.trainer.ppo.metric_utils import repeatness, check_reflection_pattern, compute_data_metrics

# Mock DataProto class for testing
class MockDataProto:
    def __init__(self, batch):
        self.batch = batch

def create_mock_batch(num_samples=2, response_length=10, prompt_length=5):
    """Create mock batch data for testing."""
    # Create mock responses - just random token IDs
    responses = torch.randint(0, 1000, (num_samples, response_length))

    # Create attention mask
    total_length = prompt_length + response_length
    attention_mask = torch.ones((num_samples, total_length))

    # Mock advantages and returns
    advantages = torch.randn((num_samples, response_length))
    returns = torch.randn((num_samples, response_length))

    # Mock token level scores and rewards
    token_level_scores = torch.randn((num_samples, response_length))
    token_level_rewards = torch.randn((num_samples, response_length))

    # Create mock values if using critic
    values = torch.randn((num_samples, response_length))

    return {
        'responses': responses,
        'attention_mask': attention_mask,
        'advantages': advantages,
        'returns': returns,
        'token_level_scores': token_level_scores,
        'token_level_rewards': token_level_rewards,
        'values': values
    }

def test_compute_data_metrics():
    """Test the compute_data_metrics function."""
    print("Testing compute_data_metrics...")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("/home/share/Qwen2.5-32B-SFT")
    except Exception as e:
        print(f"Failed to load Qwen tokenizer, trying GPT2 instead: {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create mock data
    batch_data = create_mock_batch(num_samples=3, response_length=20, prompt_length=10)
    mock_batch = MockDataProto(batch_data)

    # Generate some sample responses that would typically cause errors
    # 1. Normal response
    normal_response = "This is a normal response that should not repeat."
    # 2. Repeating response  
    repeating_response = "This is a response that repeats. This is a response that repeats."
    # 3. Response with reflection pattern
    reflection_response = "Let me rethink this. I should try again. However, I need to recheck, let me retry."

    # Map these to token IDs and replace in our mock batch
    sample_responses = [normal_response, repeating_response, reflection_response]
    encoded_responses = []
    for resp in sample_responses:
        encoded = tokenizer.encode(resp)
        # Pad or truncate to match response_length
        if len(encoded) > 20:
            encoded = encoded[:20]
        else:
            encoded = encoded + [tokenizer.pad_token_id] * (20 - len(encoded))
        encoded_responses.append(encoded)

    # Replace mock responses with our test responses
    mock_batch.batch['responses'] = torch.tensor(encoded_responses)

    # Test without tokenizer first
    print("\nTesting without tokenizer:")
    metrics_no_tokenizer = compute_data_metrics(mock_batch, use_critic=True, tokenizer=None)
    print("Repeatness metric (should be 0):", metrics_no_tokenizer.get('train/repeatness', 'Not found'))
    print("Reflection count (should be 0):", metrics_no_tokenizer.get('train/reflection_count', 'Not found'))

    # Now test with tokenizer
    print("\nTesting with tokenizer:")
    metrics_with_tokenizer = compute_data_metrics(mock_batch, use_critic=True, tokenizer=tokenizer)
    print("Repeatness metric:", metrics_with_tokenizer.get('train/repeatness', 'Not found'))
    print("Reflection count:", metrics_with_tokenizer.get('train/reflection_count', 'Not found'))

    # Test individual functions directly
    print("\nTesting repeatness function directly:")
    for resp in sample_responses:
        try:
            repeat_val = repeatness(resp)
            print(f"Repeatness for '{resp[:30]}...': {repeat_val}")
        except Exception as e:
            print(f"Error in repeatness for '{resp[:30]}...': {e}")

    print("\nTesting check_reflection_pattern function directly:")
    for resp in sample_responses:
        try:
            reflection_dict = check_reflection_pattern(resp)
            print(f"Reflection for '{resp[:30]}...': {sum(reflection_dict.values())} patterns")
            print(f"Patterns found: {dict(reflection_dict)}")
        except Exception as e:
            print(f"Error in check_reflection_pattern for '{resp[:30]}...': {e}")

    print("\nAll tests completed!")

if __name__ == "__main__":
    # First make sure the individual functions work
    print("Testing repeatness function...")
    test_str = "This is a test string."
    repeat_str = "This is a test. This is a test."

    try:
        print(f"repeatness('{test_str}'): {repeatness(test_str)}")
        print(f"repeatness('{repeat_str}'): {repeatness(repeat_str)}")
    except Exception as e:
        print(f"Error testing repeatness: {e}")
        # Check if it's due to missing imports
        print("Make sure you have these imports at the top of metric_utils.py:")
        print("from itertools import zip_longest, islice")
        print("import re")

    # Now test the compute_data_metrics function
    test_compute_data_metrics()
