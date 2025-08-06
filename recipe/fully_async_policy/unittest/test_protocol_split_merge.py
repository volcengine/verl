# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto


def create_sample_dataproto():
    """Create a DataProto similar to the provided example."""

    # Create tensor data similar to the example
    batch_size = 12

    # Tensor data
    attention_mask = torch.ones(batch_size, 3072, dtype=torch.int64)
    input_ids = torch.randint(0, 32000, (batch_size, 3072), dtype=torch.int64)
    position_ids = torch.arange(3072).unsqueeze(0).repeat(batch_size, 1).long()
    prompts = torch.randint(0, 32000, (batch_size, 1024), dtype=torch.int64)
    response_mask = torch.ones(batch_size, 2048, dtype=torch.int64)
    responses = torch.randint(0, 32000, (batch_size, 2048), dtype=torch.int64)

    # Non-tensor data similar to the example
    data_source = np.array(["openai/gsm8k"] * batch_size, dtype=object)
    ability = np.array(["math"] * batch_size, dtype=object)

    reward_model = np.array(
        [
            {"ground_truth": "6", "style": "rule"},
            {"ground_truth": "6", "style": "rule"},
            {"ground_truth": "220000", "style": "rule"},
            {"ground_truth": "277", "style": "rule"},
            {"ground_truth": "277", "style": "rule"},
            {"ground_truth": "35", "style": "rule"},
            {"ground_truth": "6", "style": "rule"},
            {"ground_truth": "220000", "style": "rule"},
            {"ground_truth": "220000", "style": "rule"},
            {"ground_truth": "277", "style": "rule"},
            {"ground_truth": "35", "style": "rule"},
            {"ground_truth": "35", "style": "rule"},
        ],
        dtype=object,
    )

    extra_info = np.array(
        [
            {"answer": "Answer 1", "index": 4570, "question": "Question 1", "split": "train"},
            {"answer": "Answer 1", "index": 4570, "question": "Question 1", "split": "train"},
            {"answer": "Answer 2", "index": 460, "question": "Question 2", "split": "train"},
            {"answer": "Answer 3", "index": 6613, "question": "Question 3", "split": "train"},
            {"answer": "Answer 3", "index": 6613, "question": "Question 3", "split": "train"},
            {"answer": "Answer 4", "index": 1421, "question": "Question 4", "split": "train"},
            {"answer": "Answer 1", "index": 4570, "question": "Question 1", "split": "train"},
            {"answer": "Answer 2", "index": 460, "question": "Question 2", "split": "train"},
            {"answer": "Answer 2", "index": 460, "question": "Question 2", "split": "train"},
            {"answer": "Answer 3", "index": 6613, "question": "Question 3", "split": "train"},
            {"answer": "Answer 4", "index": 1421, "question": "Question 4", "split": "train"},
            {"answer": "Answer 4", "index": 1421, "question": "Question 4", "split": "train"},
        ],
        dtype=object,
    )

    uid = np.array(
        [
            "80ae1835-a8db-4faa-8b42-2ffa2ca63f28",
            "80ae1835-a8db-4faa-8b42-2ffa2ca63f28",
            "cc529271-c2ba-4fe1-a16e-50c5f090538d",
            "237ea082-350f-4193-b9a2-3a153a3a38b9",
            "237ea082-350f-4193-b9a2-3a153a3a38b9",
            "fab3e910-67b3-4653-bc69-377250049267",
            "80ae1835-a8db-4faa-8b42-2ffa2ca63f28",
            "cc529271-c2ba-4fe1-a16e-50c5f090538d",
            "cc529271-c2ba-4fe1-a16e-50c5f090538d",
            "237ea082-350f-4193-b9a2-3a153a3a38b9",
            "fab3e910-67b3-4653-bc69-377250049267",
            "fab3e910-67b3-4653-bc69-377250049267",
        ],
        dtype=object,
    )

    tools_kwargs = np.array([{}] * batch_size, dtype=object)
    interaction_kwargs = np.array([{}] * batch_size, dtype=object)
    index = np.array([4570, 4570, 460, 6613, 6613, 1421, 4570, 460, 460, 6613, 1421, 1421], dtype=object)

    # Create DataProto
    data_proto = DataProto.from_dict(
        tensors={
            "attention_mask": attention_mask,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "prompts": prompts,
            "response_mask": response_mask,
            "responses": responses,
        },
        non_tensors={
            "data_source": data_source,
            "ability": ability,
            "reward_model": reward_model,
            "extra_info": extra_info,
            "uid": uid,
            "tools_kwargs": tools_kwargs,
            "interaction_kwargs": interaction_kwargs,
            "index": index,
        },
        meta_info={"global_token_num": [2141, 2141, 2161, 2151, 2151, 2130, 2141, 2161, 2161, 2151, 2130, 2130]},
    )

    return data_proto


def test_basic_split_and_merge():
    """Test basic split and merge functionality."""
    print("=== Testing Basic Split and Merge ===")

    # Create sample data
    original_proto = create_sample_dataproto()
    original_length = len(original_proto)

    print(f"Original DataProto length: {original_length}")
    print(f"Original tensor keys: {list(original_proto.batch.keys())}")
    print(f"Original non_tensor keys: {list(original_proto.non_tensor_batch.keys())}")

    # Test split
    items = original_proto.to_items()

    print(f"Split into {len(items)} items")
    assert len(items) == original_length, f"Expected {original_length} items, got {len(items)}"

    # Verify individual items
    for i, item in enumerate(items):
        print(f"Item {i}: batch_size={item.batch.batch_size}, non_tensor keys={list(item.non_tensor_batch.keys())}")

        # Check that tensor shapes are correct (no batch dimension)
        assert item.batch.batch_size == torch.Size([]), (
            f"Item {i} should have empty batch_size, got {item.batch.batch_size}"
        )

        # Check tensor shapes
        assert item.batch["attention_mask"].shape == torch.Size([3072]), (
            f"Unexpected attention_mask shape: {item.batch['attention_mask'].shape}"
        )
        assert item.batch["input_ids"].shape == torch.Size([3072]), (
            f"Unexpected input_ids shape: {item.batch['input_ids'].shape}"
        )
        assert item.batch["prompts"].shape == torch.Size([1024]), (
            f"Unexpected prompts shape: {item.batch['prompts'].shape}"
        )

        # Check non-tensor data types
        assert isinstance(item.non_tensor_batch["data_source"], str), (
            f"data_source should be str, got {type(item.non_tensor_batch['data_source'])}"
        )
        assert isinstance(item.non_tensor_batch["reward_model"], dict), (
            f"reward_model should be dict, got {type(item.non_tensor_batch['reward_model'])}"
        )
        assert isinstance(item.non_tensor_batch["extra_info"], dict), (
            f"extra_info should be dict, got {type(item.non_tensor_batch['extra_info'])}"
        )

    # Test merge
    merged_proto = DataProto.from_items(items)

    print(f"Merged DataProto length: {len(merged_proto)}")
    assert len(merged_proto) == original_length, f"Merged length should be {original_length}, got {len(merged_proto)}"

    # Verify tensor data consistency
    for key in original_proto.batch.keys():
        original_tensor = original_proto.batch[key]
        merged_tensor = merged_proto.batch[key]

        assert original_tensor.shape == merged_tensor.shape, (
            f"Shape mismatch for {key}: {original_tensor.shape} vs {merged_tensor.shape}"
        )
        assert torch.equal(original_tensor, merged_tensor), f"Tensor data mismatch for {key}"

    # Verify non-tensor data consistency
    for key in original_proto.non_tensor_batch.keys():
        original_array = original_proto.non_tensor_batch[key]
        merged_array = merged_proto.non_tensor_batch[key]

        assert original_array.shape == merged_array.shape, (
            f"Shape mismatch for {key}: {original_array.shape} vs {merged_array.shape}"
        )
        assert np.array_equal(original_array, merged_array), f"Non-tensor data mismatch for {key}"

    # Verify meta_info consistency
    assert original_proto.meta_info == merged_proto.meta_info, "Meta info mismatch"

    print("✓ Basic split and merge test passed!")


def test_individual_item_access():
    """Test accessing individual items matches split results."""
    print("\n=== Testing Individual Item Access ===")

    original_proto = create_sample_dataproto()
    items = original_proto.to_items()

    # Compare direct indexing with split results
    for i in range(len(original_proto)):
        direct_item = original_proto[i]
        split_item = items[i]

        # Check tensor data
        for key in original_proto.batch.keys():
            assert torch.equal(direct_item.batch[key], split_item.batch[key]), (
                f"Tensor mismatch at index {i}, key {key}"
            )

        # Check non-tensor data
        for key in original_proto.non_tensor_batch.keys():
            if isinstance(direct_item.non_tensor_batch[key], np.ndarray):
                assert np.array_equal(direct_item.non_tensor_batch[key], split_item.non_tensor_batch[key]), (
                    f"Non-tensor mismatch at index {i}, key {key}"
                )
            else:
                assert direct_item.non_tensor_batch[key] == split_item.non_tensor_batch[key], (
                    f"Non-tensor mismatch at index {i}, key {key}"
                )

    print("✓ Individual item access test passed!")


def test_partial_merge():
    """Test merging a subset of items."""
    print("\n=== Testing Partial Merge ===")

    original_proto = create_sample_dataproto()
    items = original_proto.to_items()

    # Take a subset of items
    subset_indices = [0, 2, 4, 7, 9]
    subset_items = [items[i] for i in subset_indices]

    # Merge the subset
    subset_proto = DataProto.from_items(subset_items)

    assert len(subset_proto) == len(subset_indices), (
        f"Subset length should be {len(subset_indices)}, got {len(subset_proto)}"
    )

    # Verify the subset contains correct data
    for i, original_idx in enumerate(subset_indices):
        # Compare with original data at original_idx
        for key in original_proto.batch.keys():
            expected_tensor = original_proto.batch[key][original_idx]
            actual_tensor = subset_proto.batch[key][i]
            assert torch.equal(expected_tensor, actual_tensor), f"Subset tensor mismatch at {i}, key {key}"

        for key in original_proto.non_tensor_batch.keys():
            expected_value = original_proto.non_tensor_batch[key][original_idx]
            actual_value = subset_proto.non_tensor_batch[key][i]

            if isinstance(expected_value, np.ndarray):
                assert np.array_equal(expected_value, actual_value), f"Subset non-tensor mismatch at {i}, key {key}"
            else:
                assert expected_value == actual_value, f"Subset non-tensor mismatch at {i}, key {key}"

    print("✓ Partial merge test passed!")


def test_item_processing():
    """Test processing individual items before merging."""
    print("\n=== Testing Item Processing ===")

    original_proto = create_sample_dataproto()
    items = original_proto.to_items()

    # Process each item (e.g., add a prefix to uid)
    processed_items = []
    for i, item in enumerate(items):
        processed_item = item.copy()  # Create a copy to avoid modifying original

        # Modify some data
        processed_item.non_tensor_batch["uid"] = f"processed_{i}_{processed_item.non_tensor_batch['uid']}"
        processed_item.non_tensor_batch["processing_step"] = i
        processed_item.meta_info["processed"] = True

        processed_items.append(processed_item)

    # Merge processed items
    processed_proto = DataProto.from_items(processed_items)

    # Verify processing was applied
    for i in range(len(processed_proto)):
        expected_uid = f"processed_{i}_{items[i].non_tensor_batch['uid']}"
        actual_uid = processed_proto.non_tensor_batch["uid"][i]
        assert actual_uid == expected_uid, (
            f"Processing failed for uid at {i}: expected {expected_uid}, got {actual_uid}"
        )

        expected_step = i
        actual_step = processed_proto.non_tensor_batch["processing_step"][i]
        assert actual_step == expected_step, (
            f"Processing step mismatch at {i}: expected {expected_step}, got {actual_step}"
        )

    #    assert processed_proto.meta_info.get("processed") == True, "Meta info processing failed"

    print("✓ Item processing test passed!")


def test_error_conditions():
    """Test error conditions."""
    print("\n=== Testing Error Conditions ===")

    # Test empty list
    try:
        DataProto.from_items([])
    except ValueError as e:
        print(f"✓ Correctly caught empty list error: {e}")

    # Test inconsistent structure
    try:
        # Create items with different tensor keys
        original_proto = create_sample_dataproto()
        items = original_proto.to_items()

        # Modify one item to have different keys
        modified_item = items[1].copy()
        modified_item.batch = TensorDict({"different_key": torch.randn(3072)}, batch_size=torch.Size([]))

        inconsistent_items = [items[0], modified_item]
        DataProto.from_items(inconsistent_items)
    except ValueError as e:
        print(f"✓ Correctly caught inconsistent structure error: {e}")

    print("✓ Error conditions test passed!")


def test_roundtrip_integrity():
    """Test multiple split/merge cycles maintain data integrity."""
    print("\n=== Testing Roundtrip Integrity ===")

    original_proto = create_sample_dataproto()
    current_proto = original_proto

    # Perform multiple split/merge cycles
    for cycle in range(3):
        print(f"Cycle {cycle + 1}")

        # Split
        items = current_proto.to_items()

        # Merge
        current_proto = DataProto.from_items(items)

        # Verify integrity
        assert len(current_proto) == len(original_proto), f"Length changed in cycle {cycle + 1}"

        for key in original_proto.batch.keys():
            assert torch.equal(original_proto.batch[key], current_proto.batch[key]), (
                f"Tensor {key} changed in cycle {cycle + 1}"
            )

        for key in original_proto.non_tensor_batch.keys():
            assert np.array_equal(original_proto.non_tensor_batch[key], current_proto.non_tensor_batch[key]), (
                f"Non-tensor {key} changed in cycle {cycle + 1}"
            )

        assert original_proto.meta_info == current_proto.meta_info, f"Meta info changed in cycle {cycle + 1}"

    print("✓ Roundtrip integrity test passed!")


def run_visual_comparison():
    """Run a visual comparison similar to the user's example."""
    print("\n=== Visual Comparison (Like User Example) ===")

    original_proto = create_sample_dataproto()

    print("Original DataProto:")
    print(f"batch_size: {original_proto.batch.batch_size}")
    print(f"tensor keys: {list(original_proto.batch.keys())}")
    print(f"non_tensor keys: {list(original_proto.non_tensor_batch.keys())}")
    print(f"Sample data_source: {original_proto.non_tensor_batch['data_source'][:3]}")
    print(f"Sample uid: {original_proto.non_tensor_batch['uid'][:3]}")

    print("\n" + "=" * 50)
    print("============= SPLIT =============")
    print("=" * 50)

    items = original_proto.to_items()

    # Show first few items
    for i in range(min(3, len(items))):
        print(f"\nDataProtoItem {i}:")
        print(f"batch_size: {items[i].batch.batch_size}")
        print(f"attention_mask shape: {items[i].batch['attention_mask'].shape}")
        print(f"input_ids shape: {items[i].batch['input_ids'].shape}")
        print(f"data_source: {items[i].non_tensor_batch['data_source']}")
        print(f"uid: {items[i].non_tensor_batch['uid']}")
        print(f"reward_model: {items[i].non_tensor_batch['reward_model']}")
        print("-" * 30)

    print("\n" + "=" * 50)
    print("============= MERGE =============")
    print("=" * 50)

    merged_proto = DataProto.from_items(items)

    print("Merged DataProto:")
    print(f"batch_size: {merged_proto.batch.batch_size}")
    print(f"tensor keys: {list(merged_proto.batch.keys())}")
    print(f"non_tensor keys: {list(merged_proto.non_tensor_batch.keys())}")
    print(f"Sample data_source: {merged_proto.non_tensor_batch['data_source'][:3]}")
    print(f"Sample uid: {merged_proto.non_tensor_batch['uid'][:3]}")

    # Verify they're identical
    success = True
    try:
        for key in original_proto.batch.keys():
            assert torch.equal(original_proto.batch[key], merged_proto.batch[key])
        for key in original_proto.non_tensor_batch.keys():
            assert np.array_equal(original_proto.non_tensor_batch[key], merged_proto.non_tensor_batch[key])
        assert original_proto.meta_info == merged_proto.meta_info
        print("\n✓ Original and merged DataProto are identical!")
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        success = False

    return success


if __name__ == "__main__":
    print("Testing DataProto Split/Merge Functionality")
    print("=" * 60)

    try:
        # Run all tests
        test_basic_split_and_merge()
        test_individual_item_access()
        test_partial_merge()
        test_item_processing()
        test_error_conditions()
        test_roundtrip_integrity()

        # Run visual comparison
        visual_success = run_visual_comparison()

        if visual_success:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED!")
            print("DataProto split/merge functionality is working correctly.")
        else:
            print("\n" + "=" * 60)
            print("❌ SOME TESTS FAILED!")

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
