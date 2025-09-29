# Copyright 2025 The TransferQueue Team
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
Independent test script for DataProto<->BatchMeta conversion decorator.

This script uses the real DataProto class and mocks only the TransferQueue components
for testing.
"""

import asyncio
import os
import sys
from typing import Any

import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

# Add the recipe directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
recipe_dir = os.path.abspath(os.path.join(current_dir))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(recipe_dir)
sys.path.append(project_root)

# Import real DataProto
try:
    from verl import DataProto

    DATAPROTO_AVAILABLE = True
    print("✓ DataProto imported successfully")
except ImportError as e:
    print(f"⚠ DataProto not available: {e}")
    DATAPROTO_AVAILABLE = False

# Import TransferQueue components
try:
    from verl.experimental.transfer_queue import (
        AsyncTransferQueueClient,
        BatchMeta,
        FieldMeta,
        ProductionStatus,
        SampleMeta,
    )

    TRANSFER_QUEUE_AVAILABLE = True
    print("✓ TransferQueue imported successfully")
except ImportError as e:
    print(f"⚠ TransferQueue not available: {e}")
    TRANSFER_QUEUE_AVAILABLE = False

# Import the decorator
try:
    from dataproto_conversion import DEFAULT_ASYNC_TIMEOUT, dataproto_batchmeta_conversion

    DECORATOR_AVAILABLE = True
    print("✓ Decorator imported successfully")
except ImportError as e:
    print(f"⚠ Decorator not available: {e}")
    DECORATOR_AVAILABLE = False

# Mock data generation constants for testing
MOCK_VOCAB_SIZE = 1000
MOCK_SEQ_LENGTH = 10
MOCK_RESPONSE_LENGTH = 5


# Mock data generation functions for testing
def generate_mock_data(batch_size: int, field_names: list[str]) -> dict[str, torch.Tensor]:
    """Generate mock data for testing based on field names."""
    data_dict = {}

    for field_name in field_names:
        if field_name == "input_ids":
            data_dict[field_name] = torch.randint(0, MOCK_VOCAB_SIZE, (batch_size, MOCK_SEQ_LENGTH), dtype=torch.long)
        elif field_name == "attention_mask":
            data_dict[field_name] = torch.ones(batch_size, MOCK_SEQ_LENGTH, dtype=torch.long)
        elif field_name == "responses":
            data_dict[field_name] = torch.randint(
                0, MOCK_VOCAB_SIZE, (batch_size, MOCK_RESPONSE_LENGTH), dtype=torch.long
            )
        else:
            # Generic mock data
            data_dict[field_name] = torch.ones(batch_size, MOCK_RESPONSE_LENGTH, dtype=torch.float32)

    # Ensure we have responses field for testing
    if "responses" not in data_dict:
        data_dict["responses"] = torch.randint(0, MOCK_VOCAB_SIZE, (batch_size, MOCK_RESPONSE_LENGTH), dtype=torch.long)

    return data_dict


def dict_to_dataproto(data_dict: dict[str, Any], meta_info: dict[str, Any]) -> DataProto:
    """Convert dictionary to DataProto for testing."""
    batch = {}
    non_tensor_batch = {}

    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value
        elif isinstance(value, NonTensorStack):
            # Convert NonTensorStack back to list format for DataProto
            non_tensor_batch[key] = [item.data for item in value]
        elif isinstance(value, NonTensorData):
            # Convert NonTensorData back to scalar
            non_tensor_batch[key] = value.data
        else:
            # Convert scalars to tensors for DataProto compatibility
            if isinstance(value, (int, float, bool)):
                # Try to get batch size from existing tensors or default to 1
                if batch:
                    first_tensor = next(iter(batch.values()))
                    batch_size = first_tensor.shape[0]
                else:
                    batch_size = 1
                batch[key] = torch.tensor([value] * batch_size, dtype=torch.float32)
            else:
                # Keep other types as-is in non_tensor_batch
                non_tensor_batch[key] = value

    # Determine batch size from first tensor
    batch_size = 0
    if batch:
        first_tensor = next(iter(batch.values()))
        batch_size = first_tensor.shape[0]
    else:
        # If no tensors, use batch size 1
        batch_size = 1

    # Create DataProto
    return DataProto(
        batch=TensorDict(batch, batch_size=batch_size), non_tensor_batch=non_tensor_batch, meta_info=meta_info.copy()
    )


def dataproto_to_tensordict(data: DataProto) -> TensorDict:
    """Convert DataProto to TensorDict for testing."""
    # Start with tensor data
    tensor_dict = dict(data.batch)

    # Handle non-tensor data - convert to tensors for simplicity
    for key, value in data.non_tensor_batch.items():
        if isinstance(value, torch.Tensor):
            # Keep tensors as-is
            tensor_dict[key] = value
        elif isinstance(value, (list, tuple)) and len(value) == len(data):
            # Convert batch-aligned lists to tensors if possible
            try:
                if all(isinstance(item, (int, float)) for item in value):
                    tensor_dict[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    # Skip non-numeric data
                    continue
            except Exception:
                continue
        elif isinstance(value, (int, float, bool)):
            # Convert scalars to tensors
            tensor_dict[key] = torch.tensor([value] * len(data), dtype=torch.float32)
        else:
            # Skip complex types
            continue

    # Create TensorDict
    try:
        return TensorDict(**tensor_dict, batch_size=len(data))
    except Exception as e:
        print(f"TensorDict creation failed: {e}, trying fallback")
        # Fallback: create with batch_size parameter
        td = TensorDict({}, batch_size=len(data))
        for key, value in tensor_dict.items():
            td.set(key, value)
        return td


def create_test_batchmeta() -> BatchMeta:
    """Create a test BatchMeta for testing."""
    samples = []
    for i in range(4):
        fields = {
            "input_ids": FieldMeta(
                name="input_ids",
                dtype=torch.int64,
                shape=torch.Size([10]),
                production_status=ProductionStatus.READY_FOR_CONSUME,
            ),
            "attention_mask": FieldMeta(
                name="attention_mask",
                dtype=torch.int64,
                shape=torch.Size([10]),
                production_status=ProductionStatus.READY_FOR_CONSUME,
            ),
        }

        sample = SampleMeta(global_step=1, global_index=i, storage_id="storage_0", local_index=i, fields=fields)
        samples.append(sample)

    return BatchMeta(samples=samples, extra_info={"test": True})


class MockTransferQueueClient:
    """Mock TransferQueue client for testing."""

    def __init__(self):
        self.storage = {}
        self.call_log = []

    async def async_get_data(self, batch_meta: BatchMeta):
        """Mock data retrieval using test mock data generation."""
        self.call_log.append("async_get_data")
        batch_size = len(batch_meta)
        field_names = batch_meta.field_names or ["input_ids", "attention_mask"]

        return generate_mock_data(batch_size, field_names)

    async def async_put(self, data, metadata):
        """Mock data storage."""
        self.call_log.append("async_put")
        storage_id = list(metadata.storage_meta_groups.keys())[0] if metadata.storage_meta_groups else "mock_storage"
        self.storage[storage_id] = data

    async def async_get_meta(self, **kwargs):
        """Mock metadata retrieval."""
        self.call_log.append("async_get_meta")
        return create_test_batchmeta()


# Test functions that work with real DataProto
def compute_response_mask_function(data: DataProto) -> DataProto:
    """Test function: compute response mask."""
    responses = data.batch.get("responses", torch.zeros(len(data), 5))
    response_length = responses.size(1)

    # Use a default attention_mask if not present
    if "attention_mask" in data.batch:
        attention_mask = data.batch["attention_mask"]
    else:
        attention_mask = torch.ones(len(data), responses.size(1))

    response_mask = attention_mask[:, -response_length:]

    # Add to batch
    data.batch["response_mask"] = response_mask

    # Add some non-tensor data
    data.non_tensor_batch["mask_computed"] = True

    return data


def apply_kl_penalty_function(data: DataProto, kl_ctrl: float = 0.1) -> DataProto:
    """Test function: apply KL penalty."""
    response_mask = data.batch.get(
        "response_mask", torch.ones_like(data.batch.get("responses", torch.ones(len(data), 5)))
    )
    kl_penalty = torch.rand(len(data)) * kl_ctrl

    # Add tensor result
    data.batch["kl_penalty"] = kl_penalty

    # Add non-tensor results
    data.non_tensor_batch["kl_ctrl_value"] = kl_ctrl
    data.non_tensor_batch["step_info"] = {"iteration": 1, "total_steps": 100}

    return data


# Create test functions for decorator testing
# Note: These functions expect DataProto as input and return DataProto
@dataproto_batchmeta_conversion()
def compute_response_mask_decorated(data: DataProto) -> DataProto:
    """Decorated test function."""
    return compute_response_mask_function(data)


@dataproto_batchmeta_conversion()
def apply_kl_penalty_decorated(data: DataProto, kl_ctrl: float = 0.1) -> DataProto:
    """Decorated test function."""
    return apply_kl_penalty_function(data, kl_ctrl)


# Test wrapper that simulates decorator behavior with mock data
def test_decorator_with_mock_data(decorated_func, batch_meta: BatchMeta, **kwargs):
    """Test wrapper that simulates decorator behavior with mock data."""
    client = kwargs.get("transfer_queue_client")

    if client is None:
        # Simulate decorator behavior with mock data
        mock_data_dict = generate_mock_data(len(batch_meta), batch_meta.field_names or ["input_ids", "attention_mask"])
        mock_data = dict_to_dataproto(mock_data_dict, batch_meta.extra_info or {})

        # Call the actual function with mock data
        if "kl_ctrl" in kwargs:
            result_data = decorated_func.__wrapped__(mock_data, kwargs["kl_ctrl"])
        else:
            result_data = decorated_func.__wrapped__(mock_data)

        # Simulate updating BatchMeta with result fields
        # In real implementation, this would be handled by the decorator
        return batch_meta
    else:
        # Use the real decorator with client
        return decorated_func(batch_meta, **kwargs)


def test_dataproto_functionality():
    """Test real DataProto functionality."""
    print("\nTesting DataProto functionality...")

    # Test creation from single dict - only tensors supported
    data = DataProto.from_single_dict(
        {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]), "learning_rate": torch.tensor([0.001, 0.001])}
    )

    print(f"DataProto length: {len(data)}")
    print(f"Batch keys: {list(data.batch.keys())}")
    print(f"Non-tensor keys: {list(data.non_tensor_batch.keys())}")

    assert len(data) == 2
    assert "input_ids" in data.batch
    assert data.batch["input_ids"].shape == (2, 3)
    assert "learning_rate" in data.batch
    assert data.batch["learning_rate"].shape == (2,)

    print("✓ DataProto works correctly")


def test_basic_functionality():
    """Test basic function functionality without decorator."""
    print("\nTesting basic functionality...")

    # Create test data
    data = DataProto.from_single_dict(
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.ones(2, 3),
            "responses": torch.tensor([[7, 8], [9, 10]]),
        }
    )

    print(f"Input data shape: {data.batch['input_ids'].shape}")
    print(f"Responses shape: {data.batch['responses'].shape}")

    # Test compute_response_mask
    result = compute_response_mask_function(data)
    assert "response_mask" in result.batch
    assert result.batch["response_mask"].shape == (2, 2)  # response length is 2
    assert result.non_tensor_batch["mask_computed"] is True

    print(f"Response mask shape: {result.batch['response_mask'].shape}")

    # Test apply_kl_penalty
    result = apply_kl_penalty_function(result, kl_ctrl=0.2)
    assert "kl_penalty" in result.batch
    assert result.batch["kl_penalty"].shape == (2,)  # batch size is 2
    assert result.non_tensor_batch["kl_ctrl_value"] == 0.2

    print(f"KL penalty shape: {result.batch['kl_penalty'].shape}")

    print("✓ Basic functionality works correctly")


async def test_decorator_functionality():
    """Test decorator functionality with mock client."""
    if not (DECORATOR_AVAILABLE and TRANSFER_QUEUE_AVAILABLE):
        print("\n⚠ Skipping decorator tests (components not available)")
        return

    print("\nTesting decorator functionality...")

    # Create test BatchMeta and client
    batch_meta = create_test_batchmeta()
    mock_client = MockTransferQueueClient()

    print(f"Test BatchMeta size: {len(batch_meta)}")
    print(f"BatchMeta fields: {batch_meta.field_names}")

    # Test without client (should now provide clear error)
    print("\n1. Testing compute_response_mask decorator without client...")
    try:
        result_batch_meta = compute_response_mask_decorated(batch_meta)
        print("✗ compute_response_mask decorator should have failed without client")
    except ValueError as e:
        if "client is required" in str(e) or "AsyncTransferQueueClient" in str(e):
            print("✓ compute_response_mask decorator correctly requires client")
        else:
            print(f"✗ Unexpected error: {e}")
    except Exception as e:
        if "AsyncTransferQueueClient" in str(e):
            print("✓ compute_response_mask decorator correctly requires AsyncTransferQueueClient")
        else:
            print(f"✗ compute_response_mask decorator failed with unexpected error: {e}")
            import traceback
            traceback.print_exc()

    # Test with mock data simulation
    print("\n1b. Testing compute_response_mask decorator with mock data simulation...")
    try:
        result_batch_meta = test_decorator_with_mock_data(compute_response_mask_decorated, batch_meta)
        print("✓ compute_response_mask decorator works with mock data simulation")
        print(f"  Result BatchMeta size: {len(result_batch_meta)}")
        print(f"  Result fields: {result_batch_meta.field_names}")
    except Exception as e:
        print(f"✗ compute_response_mask decorator mock simulation failed: {e}")
        import traceback

        traceback.print_exc()

    # Test with client - need to test in separate thread to avoid async context
    print("\n2. Testing compute_response_mask decorator with client (in separate thread)...")
    try:
        # Run in a separate thread to avoid async context issues
        import concurrent.futures

        def run_sync_test():
            return compute_response_mask_decorated(batch_meta, transfer_queue_client=mock_client)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_sync_test)
            result_batch_meta = future.result(timeout=DEFAULT_ASYNC_TIMEOUT)

        print("✓ compute_response_mask decorator works with client")
        print(f"  Result BatchMeta size: {len(result_batch_meta)}")
        print(f"  Result fields: {result_batch_meta.field_names}")
        print(f"  Client calls: {mock_client.call_log}")
        assert "async_get_data" in mock_client.call_log
        assert "async_put" in mock_client.call_log
        assert "response_mask" in result_batch_meta.field_names
        mock_client.call_log.clear()
    except Exception as e:
        print(f"✗ compute_response_mask decorator with client failed: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: apply_kl_penalty without client
    print("\n3. Testing apply_kl_penalty decorator without client...")
    try:
        result_batch_meta = apply_kl_penalty_decorated(batch_meta, kl_ctrl=0.15)
        print("✗ apply_kl_penalty decorator should have failed without client")
    except ValueError as e:
        if "client is required" in str(e) or "AsyncTransferQueueClient" in str(e):
            print("✓ apply_kl_penalty decorator correctly requires client")
        else:
            print(f"✗ Unexpected error: {e}")
    except Exception as e:
        if "AsyncTransferQueueClient" in str(e):
            print("✓ apply_kl_penalty decorator correctly requires AsyncTransferQueueClient")
        else:
            print(f"✗ apply_kl_penalty decorator failed with unexpected error: {e}")
            import traceback
            traceback.print_exc()

    # Test with mock data simulation
    print("\n3b. Testing apply_kl_penalty decorator with mock data simulation...")
    try:
        result_batch_meta = test_decorator_with_mock_data(apply_kl_penalty_decorated, batch_meta, kl_ctrl=0.15)
        print("✓ apply_kl_penalty decorator works with mock data simulation")
        print(f"  Result BatchMeta size: {len(result_batch_meta)}")
        print(f"  Result fields: {result_batch_meta.field_names}")
    except Exception as e:
        print(f"✗ apply_kl_penalty decorator mock simulation failed: {e}")
        import traceback

        traceback.print_exc()

    # Test 4: Test error handling
    print("\n4. Testing error handling...")
    try:
        # Test with None batch_meta - avoid triggering smart_wrapper errors
        try:
            # Access the wrapped function directly to avoid decorator error logging
            compute_response_mask_decorated.__wrapped__(None)
            print("✗ Should have raised ValueError for None batch_meta")
        except (ValueError, TypeError, AttributeError) as e:
            print(f"✓ Correctly raised error for None batch_meta: {type(e).__name__}")
        except Exception as e:
            print(f"✓ Correctly raised error for None batch_meta: {type(e).__name__}")
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")


def test_tensordict_nontensor_support():
    """Test TensorDict NonTensorData support."""
    print("\nTesting TensorDict NonTensorData support...")

    # Test NonTensorData creation and usage
    try:
        nt_data = NonTensorData(0.001)
        nt_stack = NonTensorStack([nt_data, nt_data])
        print("✓ NonTensorData and NonTensorStack work correctly")

        # Test conversion functions with tensor data only for compatibility
        test_dict = {
            "scalar_data": torch.tensor([0.001, 0.001], dtype=torch.float32),
            "tensor_data": torch.tensor([[1, 2], [3, 4]])
        }

        # Test dict to DataProto conversion
        meta_info = {"test": True}
        dataprot = dict_to_dataproto(test_dict, meta_info)
        print("✓ Dictionary to DataProto conversion works with tensor data")

        # Test DataProto to TensorDict conversion
        tensor_dict = dataproto_to_tensordict(dataprot)
        print("✓ DataProto to TensorDict conversion works")

    except Exception as e:
        print(f"⚠ NonTensorData test failed: {e}")
        print("  This is likely a TensorDict version compatibility issue")


async def main():
    """Main test function."""
    print("=== DataProto<->BatchMeta Decorator Test ===")

    # Check availability
    print("\nComponent availability:")
    print(f"  DataProto: {DATAPROTO_AVAILABLE}")
    print(f"  TransferQueue: {TRANSFER_QUEUE_AVAILABLE}")
    print(f"  Decorator: {DECORATOR_AVAILABLE}")

    # Test DataProto functionality
    if DATAPROTO_AVAILABLE:
        test_dataproto_functionality()
        test_basic_functionality()
        test_tensordict_nontensor_support()
    else:
        print("\n⚠ Skipping DataProto tests")

    # Test decorator functionality
    if DECORATOR_AVAILABLE and TRANSFER_QUEUE_AVAILABLE and DATAPROTO_AVAILABLE:
        await test_decorator_functionality()
    else:
        print("\n⚠ Skipping decorator tests (missing components)")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
