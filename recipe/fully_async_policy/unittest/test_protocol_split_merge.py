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

from verl.protocol import DataProto, DataProtoItem


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


def example_basic_split_merge():
    """Basic example of splitting DataProto into DataProtoItems and merging back."""
    print("=== Basic Split and Merge Example ===")

    # Create sample data
    batch_size = 3
    seq_len = 5

    # Create tensors
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Create non-tensor data
    prompts = np.array(["Hello world", "How are you?", "Good morning"], dtype=object)
    scores = np.array([0.8, 0.9, 0.7], dtype=object)

    # Create DataProto
    data_proto = DataProto.from_dict(
        tensors={"input_ids": input_ids, "attention_mask": attention_mask},
        non_tensors={"prompts": prompts, "scores": scores},
        meta_info={"model_name": "test_model", "version": "1.0"},
    )

    print(f"Original DataProto length: {len(data_proto)}")
    print(f"Input IDs shape: {data_proto.batch['input_ids'].shape}")
    print(f"Prompts: {data_proto.non_tensor_batch['prompts']}")

    # Split into DataProtoItems
    items = data_proto.to_items()
    print(f"\nSplit into {len(items)} items")

    for i, item in enumerate(items):
        print(f"Item {i}:")
        print(f"  Input IDs shape: {item.batch['input_ids'].shape}")
        print(f"  Prompt: {item.non_tensor_batch['prompts']}")
        print(f"  Score: {item.non_tensor_batch['scores']}")

    # Merge back to DataProto
    merged_proto = DataProto.from_items(items)
    print(f"\nMerged DataProto length: {len(merged_proto)}")
    print(f"Merged Input IDs shape: {merged_proto.batch['input_ids'].shape}")
    print(f"Merged prompts: {merged_proto.non_tensor_batch['prompts']}")

    # Verify they're identical
    assert torch.equal(data_proto.batch["input_ids"], merged_proto.batch["input_ids"])
    assert torch.equal(data_proto.batch["attention_mask"], merged_proto.batch["attention_mask"])
    assert np.array_equal(data_proto.non_tensor_batch["prompts"], merged_proto.non_tensor_batch["prompts"])
    assert np.array_equal(data_proto.non_tensor_batch["scores"], merged_proto.non_tensor_batch["scores"])

    print("\n✓ Original and merged DataProto are identical!")


def example_item_processing():
    """Example showing individual item processing before merging."""
    print("\n=== Individual Item Processing Example ===")

    # Create initial data
    #    batch_size = 4

    values = torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(1)  # Shape: (4, 1)
    labels = np.array(["A", "B", "C", "D"], dtype=object)

    original_proto = DataProto.from_dict(
        tensors={"values": values}, non_tensors={"labels": labels}, meta_info={"processing_step": 0}
    )

    print(f"Original values: {original_proto.batch['values'].flatten()}")
    print(f"Original labels: {original_proto.non_tensor_batch['labels']}")

    # Split and process each item individually
    items = original_proto.to_items()
    processed_items = []

    for i, item in enumerate(items):
        # Process the tensor data (multiply by 2)
        processed_value = item.batch["values"] * 2

        # Process the non-tensor data (add suffix)
        processed_label = item.non_tensor_batch["labels"] + f"_processed_{i}"

        # Create new processed item
        processed_item = DataProtoItem(
            batch=item.batch.clone(),  # Clone the TensorDict
            non_tensor_batch=item.non_tensor_batch.copy(),
            meta_info=item.meta_info.copy(),
        )

        # Update with processed data
        processed_item.batch["values"] = processed_value
        processed_item.non_tensor_batch["labels"] = processed_label
        processed_item.meta_info["processing_step"] = 1

        processed_items.append(processed_item)

        print(f"Processed item {i}: value={processed_value.item()}, label='{processed_label}'")

    # Merge processed items back
    processed_proto = DataProto.from_items(processed_items)

    print(f"\nProcessed values: {processed_proto.batch['values'].flatten()}")
    print(f"Processed labels: {processed_proto.non_tensor_batch['labels']}")
    print(f"Processing step: {processed_proto.meta_info['processing_step']}")


def example_convenience_methods():
    """Example showing convenience methods."""
    print("\n=== Convenience Methods Example ===")

    # Create a single DataProtoItem
    single_tensor = torch.tensor([42]).unsqueeze(0)  # Shape: (1,)
    single_item = DataProtoItem(
        batch=None,  # We'll create TensorDict manually
        non_tensor_batch={"text": "Hello"},
        meta_info={"source": "manual"},
    )

    # Create TensorDict manually for the single item
    from tensordict import TensorDict

    single_item.batch = TensorDict({"data": single_tensor}, batch_size=(1,))

    print(f"Single item data: {single_item.batch['data']}")
    print(f"Single item text: {single_item.non_tensor_batch['text']}")

    # Convert single item to DataProto using convenience method
    single_proto = single_item.to_proto()
    print(f"Converted to DataProto length: {len(single_proto)}")

    # Create multiple items and use static convenience method
    items = [single_item]
    for i in range(2):
        new_item = single_item.copy()  # Use the copy method
        new_item.batch["data"] = torch.tensor([100 + i]).unsqueeze(0)
        new_item.non_tensor_batch["text"] = f"Item {i + 1}"
        items.append(new_item)

    # Use DataProtoItem.from_items() convenience method
    merged_proto = DataProtoItem.from_items(items)
    print(f"Merged using convenience method - length: {len(merged_proto)}")
    print(f"Data: {merged_proto.batch['data'].flatten()}")
    print(f"Texts: {merged_proto.non_tensor_batch['text']}")


def example_error_handling():
    """Example showing error handling."""
    print("\n=== Error Handling Example ===")

    # Try to create DataProto from empty list
    try:
        DataProto.from_items([])
        print("ERROR: Should have raised exception for empty list")
    except ValueError as e:
        print(f"✓ Correctly caught error for empty list: {e}")

    # Try to merge items with inconsistent structure
    try:
        item1 = DataProtoItem(
            batch=TensorDict({"data": torch.tensor([1]).unsqueeze(0)}, batch_size=(1,)),
            non_tensor_batch={"text": "Hello"},
        )
        item2 = DataProtoItem(
            batch=TensorDict({"different_key": torch.tensor([2]).unsqueeze(0)}, batch_size=(1,)),
            non_tensor_batch={"text": "World"},
        )

        DataProto.from_items([item1, item2])
        print("ERROR: Should have raised exception for inconsistent structure")
    except ValueError as e:
        print(f"✓ Correctly caught error for inconsistent structure: {e}")


if __name__ == "__main__":
    # Run all tests
    test_basic_split_and_merge()
    test_individual_item_access()
    test_partial_merge()
    test_item_processing()
    test_error_conditions()
    test_roundtrip_integrity()
    example_basic_split_merge()
    example_item_processing()
    example_convenience_methods()
    example_error_handling()
    run_visual_comparison()
