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

from verl.protocol import DataProto, DataProtoItem


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
    # Import tensordict for the examples
    from tensordict import TensorDict

    # Run all examples
    example_basic_split_merge()
    example_item_processing()
    example_convenience_methods()
    example_error_handling()

    print("\n🎉 All examples completed successfully!")
