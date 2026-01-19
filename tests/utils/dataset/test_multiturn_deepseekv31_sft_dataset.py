# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates

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
from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer

from verl.utils.dataset.multiturn_sft_dataset_deepseek_v31 import MultiTurnSFTDatasetDeepseek

custom_model_prefix = Path("~/models").expanduser().resolve()


# This test is performed under the chat template provided by sglang
# which accpet and uses the tool provided in the chat template
@pytest.mark.parametrize(
    "model_path",
    [
        f"{custom_model_prefix}/deepseek-ai/DeepSeek-V3.1",
    ],
)
@pytest.mark.parametrize("enable_thinking", [False, True])
def test_multiturn_sft_dataset(model_path: str, enable_thinking: bool):
    print(f"Starting test... model_path={model_path}, enable_thinking={enable_thinking}")
    # Create a temporary parquet file with test data
    test_data = {
        "messages": [
            [
                {"role": "system", "content": "SYSTEMPROMPT\n"},
                {"role": "user", "content": "USERPROMPT\n"},
                {
                    "role": "assistant",
                    "content": "<think>THINK</think>ASSISTANTPROMPT WITHTOOL\n",
                    "tool_calls": [{"function": {"name": "\nTOOLNAME", "arguments": "ARGUMENT"}}],
                },
                {"role": "tool", "content": "TOOLCALL 1\n"},
                {
                    "role": "assistant",
                    "content": "<think>THINK2</think>ANSWER\n",
                    "tool_calls": [{"function": {"name": "\nTOOLNAME2", "arguments": "ARGUMENT2"}}],
                },
                {"role": "tool", "content": "TOOLCALL 2\n"},
                {"role": "assistant", "content": "<think>THINK3</think>ANSWER\n"},
            ],
        ],
        "tools": [
            [
                {
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {"location": "string", "date": "string"},
                    }
                }
            ]
        ],
    }

    # Create test directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    test_file = "test_data/test.parquet"

    # Save test data to parquet
    df = pd.DataFrame(test_data)
    df.to_parquet(test_file)

    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = {
        "max_length": 512,
        "truncation": "error",
        "multiturn": {"messages_key": "messages"},
        "apply_chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    dataset = MultiTurnSFTDatasetDeepseek(parquet_files=test_file, tokenizer=tokenizer, config=config)

    # Test 1: Dataset Length
    assert len(dataset) == len(test_data["messages"]), f"Expected dataset length 2, got {len(dataset)}"

    # Get items for testing
    item0 = dataset[0]  # multi-turn With tool
    # Multiple chats turn
    # item1 = dataset[1]

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
    small_dataset = MultiTurnSFTDatasetDeepseek(parquet_files=test_file, tokenizer=tokenizer, config=padding_config)
    padded_item = small_dataset[0]

    # Get actual sequence length (before padding)
    actual_length = torch.sum(padded_item["attention_mask"])

    # Verify padding tokens
    assert torch.all(padded_item["input_ids"][actual_length:] == tokenizer.pad_token_id), (
        "Padding tokens not set correctly"
    )
    assert torch.all(padded_item["attention_mask"][actual_length:] == 0), "Attention mask not set correctly for padding"
    assert torch.all(padded_item["loss_mask"][actual_length:] == 0), "Loss mask not set correctly for padding"

    print("All tests passed!")
    print("Starting test...")
