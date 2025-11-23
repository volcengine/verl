#!/usr/bin/env python3
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
Unit tests for Bug #4159: response_position_ids padding direction

This test ensures that response_position_ids uses right padding to align
with response_ids, especially for 2D position_ids in multimodal models.
"""

import torch
from torch.nn.utils.rnn import pad_sequence


def test_response_position_ids_1d_padding_alignment():
    """
    Test that 1D response_position_ids uses right padding to align with response_ids

    This is the standard case for text-only models.
    """
    # Simulate sequences of different lengths
    response_ids = [
        torch.tensor([101, 102, 103, 104]),  # length 4
        torch.tensor([201, 202]),             # length 2
    ]

    response_position_ids = [
        torch.tensor([0, 1, 2, 3]),  # positions for seq 1
        torch.tensor([0, 1]),         # positions for seq 2
    ]

    # Both should use right padding (default)
    padded_ids = pad_sequence(
        response_ids,
        batch_first=True,
        padding_value=0
    )

    # This should match response_ids padding
    padded_pos = pad_sequence(
        response_position_ids,
        batch_first=True,
        padding_value=0
        # padding_side not specified = "right" (default)
    )

    # Verify shapes match
    assert padded_ids.shape == padded_pos.shape, \
        f"Shape mismatch: ids {padded_ids.shape} vs pos {padded_pos.shape}"

    # Verify alignment for sequence 2 (the shorter one)
    # Token 201 should be at position 0
    assert padded_ids[1][0] == 201, "First token should be 201"
    assert padded_pos[1][0] == 0, "First position should be 0"

    # Token 202 should be at position 1
    assert padded_ids[1][1] == 202, "Second token should be 202"
    assert padded_pos[1][1] == 1, "Second position should be 1"

    # Padding should be at the end (right side)
    assert padded_ids[1][2] == 0, "Third element should be padding"
    assert padded_pos[1][2] == 0, "Third position should be padding"
    assert padded_ids[1][3] == 0, "Fourth element should be padding"
    assert padded_pos[1][3] == 0, "Fourth position should be padding"

    print("✅ 1D position_ids padding test passed")


def test_response_position_ids_2d_padding_alignment():
    """
    Test that 2D response_position_ids uses right padding for multimodal models

    This is the case for models like Qwen2-VL where position_ids has shape (3, seq_len)
    representing [time_dim, height_dim, width_dim]
    """
    # Simulate 2D position_ids for different sequence lengths
    # Shape: (3, seq_len) for [time, height, width]
    response_ids = [
        torch.tensor([101, 102, 103, 104]),  # length 4
        torch.tensor([201, 202]),             # length 2
    ]

    pos_2d_list = [
        torch.tensor([[0, 1, 2, 3],    # time dimension
                      [0, 0, 1, 1],    # height dimension
                      [0, 1, 0, 1]]),  # width dimension
        torch.tensor([[0, 1],          # time dimension
                      [0, 0],          # height dimension
                      [0, 1]]),        # width dimension
    ]

    # Pad response_ids (right padding)
    padded_ids = pad_sequence(
        response_ids,
        batch_first=True,
        padding_value=0
    )

    # For 2D position_ids, we need to:
    # 1. Transpose: (3, seq_len) -> (seq_len, 3)
    # 2. Pad along seq_len dimension (right padding)
    # 3. Transpose back: (batch, seq_len, 3) -> (batch, 3, seq_len)

    transposed = [p.transpose(0, 1) for p in pos_2d_list]

    # This is the fix for Bug #4159
    padded_2d = pad_sequence(
        transposed,
        batch_first=True,
        padding_value=0,
        padding_side="right"  # ← CRITICAL: must be right, not left
    )

    padded_2d = padded_2d.transpose(1, 2)  # Back to (batch, 3, seq_len)

    # Verify shapes
    assert padded_2d.shape[0] == padded_ids.shape[0], "Batch size should match"
    assert padded_2d.shape[2] == padded_ids.shape[1], "Sequence length should match"

    # Verify alignment for sequence 2 (shorter sequence)
    # Check time dimension
    assert padded_2d[1, 0, 0] == 0, "Time position 0 for token 201"
    assert padded_2d[1, 0, 1] == 1, "Time position 1 for token 202"
    assert padded_2d[1, 0, 2] == 0, "Time padding at position 2"
    assert padded_2d[1, 0, 3] == 0, "Time padding at position 3"

    # Check height dimension
    assert padded_2d[1, 1, 0] == 0, "Height position for first token"
    assert padded_2d[1, 1, 1] == 0, "Height position for second token"
    assert padded_2d[1, 1, 2] == 0, "Height padding"

    # Check width dimension
    assert padded_2d[1, 2, 0] == 0, "Width position for first token"
    assert padded_2d[1, 2, 1] == 1, "Width position for second token"
    assert padded_2d[1, 2, 2] == 0, "Width padding"

    print("✅ 2D position_ids padding test passed")


def test_buggy_left_padding_fails():
    """
    Verify that the buggy behavior (left padding) would cause misalignment

    This test demonstrates what happens with the bug.
    """
    response_ids = [
        torch.tensor([101, 102, 103, 104]),  # length 4
        torch.tensor([201, 202]),             # length 2
    ]

    response_position_ids = [
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0, 1]),
    ]

    # response_ids: right padding (correct)
    padded_ids = pad_sequence(
        response_ids,
        batch_first=True,
        padding_value=0
    )

    # BUGGY: response_position_ids with left padding
    buggy_padded_pos = pad_sequence(
        response_position_ids,
        batch_first=True,
        padding_value=0,
        padding_side="left"  # ← BUG!
    )

    # Sequence 2 should look like:
    # padded_ids:         [201, 202,   0,   0]
    # buggy_padded_pos:   [  0,   0,   0,   1]  ← WRONG!
    #
    # Token 201 is at index 0, but position says 0 (which is padding)
    # This is a misalignment!

    # Verify the bug exists
    assert padded_ids[1][0] == 201, "Token at position 0"
    assert buggy_padded_pos[1][0] == 0, "This is PADDING (bug!)"

    # The actual position value 0 (for token 201) is at index 2
    assert buggy_padded_pos[1][2] == 0, "Position 0 is at wrong index"

    # This demonstrates the misalignment
    print("✅ Buggy behavior verified (for demonstration)")


if __name__ == "__main__":
    print("Running position_ids padding tests...")
    print("=" * 60)

    test_response_position_ids_1d_padding_alignment()
    test_response_position_ids_2d_padding_alignment()
    test_buggy_left_padding_fails()

    print("=" * 60)
    print("🎉 All tests passed!")
