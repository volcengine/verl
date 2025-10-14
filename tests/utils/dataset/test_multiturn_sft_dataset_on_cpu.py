# Copyright 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test the MultiTurnSFTDataset implementation
"""

import os

import pandas as pd
import torch
from transformers import AutoTokenizer

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def test_multiturn_sft_dataset():
    print("Starting test...")
    # Create a temporary parquet file with test data
    test_data = {
        "messages": [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "And what is 4+4?"},
                {"role": "assistant", "content": "4+4 equals 8."},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "Why?"},
                {"role": "assistant", "content": "To get to the other side!"},
            ],
        ]
    }

    # Create test directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    test_file = "test_data/test.parquet"

    # Save test data to parquet
    df = pd.DataFrame(test_data)
    df.to_parquet(test_file)

    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    config = {"max_length": 512, "truncation": "error", "multiturn": {"messages_key": "messages"}}
    dataset = MultiTurnSFTDataset(parquet_files=test_file, tokenizer=tokenizer, config=config)

    # Test 1: Dataset Length
    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"

    # Get items for testing
    item0 = dataset[0]  # Math conversation
    item1 = dataset[1]  # Joke conversation

    # Test 2: Required Keys and Types
    required_keys = ["input_ids", "attention_mask", "position_ids", "loss_mask"]
    for key in required_keys:
        assert key in item0, f"Missing key {key} in dataset item"
        assert isinstance(item0[key], torch.Tensor), f"Expected torch.Tensor for {key}"
        assert item0[key].dtype == torch.long, f"Expected torch.long for {key}, got {item0[key].dtype}"

    # Test 3: Shape Consistency
    assert item0["loss_mask"].shape == item0["input_ids"].shape, "Loss mask shape doesn't match input_ids shape"
    assert item0["attention_mask"].shape == item0["input_ids"].shape, (
        "Attention mask shape doesn't match input_ids shape"
    )
    assert item0["position_ids"].shape == item0["input_ids"].shape, "Position IDs shape doesn't match input_ids shape"

    # Test 4: Loss Mask Pattern - Math Conversation
    loss_mask0 = item0["loss_mask"]
    input_ids0 = item0["input_ids"]

    # Find assistant response positions
    assistant_positions0 = torch.where(loss_mask0 == 1)[0]
    assert len(assistant_positions0) > 0, "No assistant positions found in loss mask"

    # Decode and verify assistant responses
    assistant_text0 = tokenizer.decode(input_ids0[loss_mask0 == 1])
    print(f"Math conversation assistant text: {assistant_text0}")
    assert "2+2 equals 4" in assistant_text0, "First assistant response not found"
    assert "4+4 equals 8" in assistant_text0, "Second assistant response not found"

    # Test 5: Loss Mask Pattern - Joke Conversation
    loss_mask1 = item1["loss_mask"]
    input_ids1 = item1["input_ids"]

    # Find assistant response positions
    assistant_positions1 = torch.where(loss_mask1 == 1)[0]
    assert len(assistant_positions1) > 0, "No assistant positions found in loss mask"

    # Decode and verify assistant responses
    assistant_text1 = tokenizer.decode(input_ids1[loss_mask1 == 1])
    print(f"Joke conversation assistant text: {assistant_text1}")
    assert "chicken cross the road" in assistant_text1, "First assistant response not found"
    assert "other side" in assistant_text1, "Second assistant response not found"

    # Test 6: Attention Mask Pattern
    attention_mask0 = item0["attention_mask"]
    sequence_length = torch.sum(attention_mask0)
    assert sequence_length > 0, "No tokens marked as attended in attention mask"
    assert torch.all(attention_mask0[:sequence_length] == 1), "Incorrect attention mask pattern"
    if sequence_length < len(attention_mask0):
        assert torch.all(attention_mask0[sequence_length:] == 0), "Padding not properly masked"

    # Test 7: Position IDs Pattern
    position_ids0 = item0["position_ids"]
    assert torch.equal(position_ids0[:sequence_length], torch.arange(sequence_length)), (
        "Position IDs not sequential for non-padded tokens"
    )
    if sequence_length < len(position_ids0):
        assert torch.all(position_ids0[sequence_length:] == 0), "Padding position IDs not zero"

    # Test 8: Verify loss mask for assistant responses
    # Get the full conversation text
    full_text = tokenizer.decode(input_ids0)
    print(f"\nFull conversation text:\n{full_text}")

    # Get the assistant responses
    assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 1])
    print(f"\nAssistant responses (from loss mask):\n{assistant_text}")

    # Verify that loss mask is set for all assistant responses
    for msg in test_data["messages"][0]:  # First conversation
        if msg["role"] == "assistant":
            # The content should appear in the masked text
            assert msg["content"] in assistant_text, f"Assistant message '{msg['content']}' not found in masked text"

            # The content should NOT appear in the non-masked text
            non_assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 0])
            assert msg["content"] not in non_assistant_text, (
                f"Assistant message '{msg['content']}' found in non-assistant text"
            )

    # Test 9: Verify non-assistant parts have loss_mask=0
    # Get non-assistant text
    non_assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 0])
    print(f"\nNon-assistant text (from loss mask):\n{non_assistant_text}")

    # Verify that system and user messages are in the non-assistant text
    for msg in test_data["messages"][0]:  # First conversation
        if msg["role"] in ["system", "user"]:
            assert msg["content"] in non_assistant_text, (
                f"{msg['role'].title()} message '{msg['content']}' not found in non-assistant text"
            )

            # And verify they're NOT in the assistant text
            assert msg["content"] not in assistant_text, (
                f"{msg['role'].title()} message '{msg['content']}' found in assistant text"
            )

    # Test 10: Verify padding behavior
    padding_config = {"max_length": 1024, "truncation": "error", "multiturn": {"messages_key": "messages"}}
    small_dataset = MultiTurnSFTDataset(parquet_files=test_file, tokenizer=tokenizer, config=padding_config)
    padded_item = small_dataset[0]

    # Get actual sequence length (before padding)
    actual_length = torch.sum(padded_item["attention_mask"])

    # Verify padding tokens
    assert torch.all(padded_item["input_ids"][actual_length:] == tokenizer.pad_token_id), (
        "Padding tokens not set correctly"
    )
    assert torch.all(padded_item["attention_mask"][actual_length:] == 0), "Attention mask not set correctly for padding"
    assert torch.all(padded_item["loss_mask"][actual_length:] == 0), "Loss mask not set correctly for padding"

    # test no-padding
    config = {
        "max_length": 512,
        "truncation": "error",
        "multiturn": {"messages_key": "messages"},
        "pad_mode": "no_padding",
    }
    dataset = MultiTurnSFTDataset(parquet_files=test_file, tokenizer=tokenizer, config=config)

    item0 = dataset[0]

    # Verify that the output contains expected keys for no-padding mode
    required_keys = ["input_ids", "position_ids", "loss_mask"]
    for key in required_keys:
        assert key in item0, f"Missing key {key} in no-padding mode dataset item"
        assert isinstance(item0[key], torch.Tensor), f"Expected torch.Tensor for {key} in no-padding mode"

    # make sure assistant_text matches with expected
    assistant_text = tokenizer.decode(item0["input_ids"][item0["loss_mask"] == 1])
    assert assistant_text == "2+2 equals 4.<|im_end|>\n4+4 equals 8.<|im_end|>\n"

    print("All tests passed!")
    print("Starting test...")
