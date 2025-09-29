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
DataProto conversion decorator for TransferQueue integration.

This module provides a decorator that enables DataProto-based functions to work
with TransferQueue's BatchMeta system. The decorator handles the conversion
between DataProto and BatchMeta formats seamlessly.

Pattern:
1. Input: BatchMeta + TransferQueueClient
2. Decorator: BatchMeta -> DataProto -> function(DataProto) -> DataProto -> update BatchMeta
3. Output: Updated BatchMeta

Usage:
    @dataproto_batchmeta_conversion(client)
    def apply_kl_penalty(data: DataProto, kl_ctrl) -> DataProto:
        response_mask = data.batch["response_mask"]
        # ... compute kl_penalty ...
        data.batch["kl_penalty"] = kl_penalty_result
        return data

    # Usage with BatchMeta:
    batch_meta = apply_kl_penalty(batch_meta, kl_ctrl, transfer_queue_client=client)
"""

import asyncio
import functools
import logging
from typing import Any, Callable, Optional

import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl import DataProto
from verl.experimental.transfer_queue import AsyncTransferQueueClient, BatchMeta

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_ASYNC_TIMEOUT = 10.0


def dataproto_batchmeta_conversion(
    transfer_queue_client: Optional[AsyncTransferQueueClient] = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator factory for converting DataProto functions to work with BatchMeta.

    This decorator enables DataProto-based functions to work with TransferQueue's
    BatchMeta system by:
    1. Converting BatchMeta input to DataProto via client
    2. Calling the wrapped function with DataProto
    3. Converting function's DataProto output back to update BatchMeta
    4. Returning the updated BatchMeta

    Args:
        transfer_queue_client: AsyncTransferQueueClient for data operations

    Returns:
        A decorator function that wraps the target function

    Raises:
        RuntimeError: When sync function is called with async client in running event loop

    Example:
        @dataproto_batchmeta_conversion(client)
        def apply_kl_penalty(data: DataProto, kl_ctrl: float) -> DataProto:
            response_mask = data.batch["response_mask"]
            kl_penalty = torch.rand(len(data)) * kl_ctrl
            data.batch["kl_penalty"] = kl_penalty
            return data

        # Usage with BatchMeta:
        result_meta = apply_kl_penalty(batch_meta, 0.1, transfer_queue_client=client)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> BatchMeta:
            """Async wrapper for DataProto functions."""
            try:
                # Extract batch_meta and client from arguments
                batch_meta, client, other_args, other_kwargs = _extract_args(args, kwargs, transfer_queue_client)

                if batch_meta is None:
                    raise ValueError("batch_meta cannot be None")

                # Convert BatchMeta to DataProto
                data = await _batchmeta_to_dataproto_async(batch_meta, client)

                # Call function with DataProto - handle both sync and async functions
                if asyncio.iscoroutinefunction(func):
                    result_data = await func(data, *other_args, **other_kwargs)
                else:
                    result_data = func(data, *other_args, **other_kwargs)

                # Validate result
                if not isinstance(result_data, DataProto):
                    raise TypeError(f"Function {func.__name__} must return DataProto, got {type(result_data)}")

                # Update BatchMeta with result
                await _update_batchmeta_with_result_async(result_data, batch_meta, client)

                return batch_meta
            except Exception as e:
                logger.error(f"Error in async_wrapper for {func.__name__}: {e}")
                raise

        @functools.wraps(func)
        def smart_wrapper(*args: Any, **kwargs: Any) -> BatchMeta:
            """Smart wrapper that detects async context and handles appropriately."""
            try:
                # Extract batch_meta and client from arguments
                batch_meta, client, other_args, other_kwargs = _extract_args(args, kwargs, transfer_queue_client)

                if batch_meta is None:
                    raise ValueError("batch_meta cannot be None")

                # No client support for sync wrapper - require async wrapper with client
                if client is None:
                    raise ValueError(
                        "Sync wrapper requires an AsyncTransferQueueClient. "
                        "Either provide a client or use an async function with an async wrapper."
                    )

                # Handle async client in sync context
                return _handle_sync_with_async_client(batch_meta, client, other_args, other_kwargs, async_wrapper)
            except Exception as e:
                logger.error(f"Error in smart_wrapper for {func.__name__}: {e}")
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else smart_wrapper

    return decorator


def _extract_args(
    args: tuple, kwargs: dict, default_client: Optional[AsyncTransferQueueClient]
) -> tuple[Optional[BatchMeta], Optional[AsyncTransferQueueClient], tuple, dict]:
    """
    Extract batch_meta, client, and other arguments from function call.

    Args:
        args: Positional arguments from function call
        kwargs: Keyword arguments from function call
        default_client: Default client to use if none provided

    Returns:
        Tuple of (batch_meta, client, other_args, other_kwargs)

    Note:
        This function modifies kwargs by removing 'transfer_queue_client' if present
    """
    if not args:
        logger.warning("No arguments provided to decorated function")
        return None, default_client, (), kwargs.copy()

    # Find batch_meta (first argument)
    batch_meta = args[0]
    if not isinstance(batch_meta, BatchMeta):
        raise TypeError(f"First argument must be BatchMeta, got {type(batch_meta)}")

    # Find client in kwargs or use default
    client = kwargs.pop("transfer_queue_client", default_client)

    # Remaining arguments
    other_args = args[1:] if len(args) > 1 else ()
    other_kwargs = kwargs.copy()

    return batch_meta, client, other_args, other_kwargs


def _batchmeta_to_dataproto_sync(batch_meta: BatchMeta, client: Optional[AsyncTransferQueueClient]) -> DataProto:
    """
    Convert BatchMeta to DataProto (synchronous).

    Args:
        batch_meta: BatchMeta to convert
        client: Optional async client for data retrieval

    Returns:
        DataProto containing the converted data

    Raises:
        RuntimeError: If called when an event loop is running
        ValueError: If batch_meta is invalid or client is None
    """
    if not batch_meta:
        raise ValueError("batch_meta cannot be None or empty")

    if client is None:
        raise ValueError("client is required for DataProto conversion")

    # For sync wrapper, we need to handle async client carefully
    try:
        asyncio.get_running_loop()
        # We're in a running event loop in this thread; cannot safely run coroutine synchronously
        raise RuntimeError(
            "Cannot call _batchmeta_to_dataproto_sync when an event loop is running in this thread. "
            "Use the async version (_batchmeta_to_dataproto_async) instead."
        )
    except RuntimeError as e:
        if "no running event loop" in str(e):
            # No running loop, we can use asyncio.run
            data_dict = asyncio.run(client.async_get_data(batch_meta))
        else:
            raise

    return _dict_to_dataproto(data_dict, batch_meta.extra_info or {})


async def _batchmeta_to_dataproto_async(batch_meta: BatchMeta, client: Optional[AsyncTransferQueueClient]) -> DataProto:
    """
    Convert BatchMeta to DataProto (asynchronous).

    Args:
        batch_meta: BatchMeta to convert
        client: Async client for data retrieval

    Returns:
        DataProto containing the converted data

    Raises:
        ValueError: If batch_meta is invalid or client is None
        asyncio.TimeoutError: If client operation times out
    """
    if not batch_meta:
        raise ValueError("batch_meta cannot be None or empty")

    if client is None:
        raise ValueError("client is required for DataProto conversion")

    # Get data from storage with timeout
    try:
        data_dict = await asyncio.wait_for(client.async_get_data(batch_meta), timeout=DEFAULT_ASYNC_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error(f"Timeout getting data from client for batch_meta with {len(batch_meta)} samples")
        raise

    return _dict_to_dataproto(data_dict, batch_meta.extra_info or {})


def _update_batchmeta_with_result_sync(
    result_data: DataProto, batch_meta: BatchMeta, client: Optional[AsyncTransferQueueClient]
) -> None:
    """
    Update BatchMeta with DataProto result (synchronous).

    Args:
        result_data: DataProto result to convert and store
        batch_meta: BatchMeta to update with new fields
        client: Optional async client for data storage

    Raises:
        RuntimeError: If called when an event loop is running
        ValueError: If inputs are invalid
    """
    if not result_data:
        raise ValueError("result_data cannot be None or empty")
    if not batch_meta:
        raise ValueError("batch_meta cannot be None or empty")

    # Convert DataProto to TensorDict
    output_tensor_dict = _dataproto_to_tensordict(result_data)

    if client is not None:
        # Store output data
        try:
            asyncio.get_running_loop()
            # We're in a running event loop in this thread; cannot safely run coroutine synchronously
            raise RuntimeError(
                "Cannot call _update_batchmeta_with_result_sync when an event loop is running in this thread. "
                "Use the async version (_update_batchmeta_with_result_async) instead."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e):
                # No running loop, we can use asyncio.run
                asyncio.run(
                    asyncio.wait_for(
                        client.async_put(data=output_tensor_dict, metadata=batch_meta), timeout=DEFAULT_ASYNC_TIMEOUT
                    )
                )
            else:
                raise

    # Update BatchMeta with new fields
    try:
        batch_meta.add_fields(output_tensor_dict)
    except Exception as e:
        logger.error(f"Failed to update BatchMeta with new fields: {e}")
        raise


async def _update_batchmeta_with_result_async(
    result_data: DataProto, batch_meta: BatchMeta, client: Optional[AsyncTransferQueueClient]
) -> None:
    """
    Update BatchMeta with DataProto result (asynchronous).

    Args:
        result_data: DataProto result to convert and store
        batch_meta: BatchMeta to update with new fields
        client: Optional async client for data storage

    Raises:
        asyncio.TimeoutError: If client operation times out
        ValueError: If inputs are invalid
    """
    if not result_data:
        raise ValueError("result_data cannot be None or empty")
    if not batch_meta:
        raise ValueError("batch_meta cannot be None or empty")

    # Convert DataProto to TensorDict
    output_tensor_dict = _dataproto_to_tensordict(result_data)

    if client is not None:
        # Store output data with timeout
        try:
            await asyncio.wait_for(
                client.async_put(data=output_tensor_dict, metadata=batch_meta), timeout=DEFAULT_ASYNC_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout storing data to client for batch_meta with {len(batch_meta)} samples")
            raise

    # Update BatchMeta with new fields
    try:
        batch_meta.add_fields(output_tensor_dict)
    except Exception as e:
        logger.error(f"Failed to update BatchMeta with new fields: {e}")
        raise


def _dict_to_dataproto(data_dict: dict[str, Any], meta_info: dict[str, Any]) -> DataProto:
    """
    Convert dictionary to DataProto, handling NonTensorData.

    Args:
        data_dict: Dictionary containing tensor and non-tensor data
        meta_info: Metadata information for DataProto

    Returns:
        DataProto containing the converted data

    Raises:
        ValueError: If data_dict is empty or invalid
        TypeError: If data types are unsupported
    """
    if not data_dict:
        raise ValueError("data_dict cannot be empty")

    batch = {}
    non_tensor_batch = {}

    for key, value in data_dict.items():
        if not isinstance(key, str):
            raise TypeError(f"Key must be string, got {type(key)}")

        try:
            if isinstance(value, torch.Tensor):
                batch[key] = value
            elif isinstance(value, NonTensorStack):
                # Convert NonTensorStack back to list format for DataProto
                non_tensor_batch[key] = [item.data for item in value]
            elif isinstance(value, NonTensorData):
                # Convert NonTensorData back to scalar
                non_tensor_batch[key] = value.data
            else:
                # Keep other types as-is
                non_tensor_batch[key] = value
        except Exception as e:
            logger.warning(f"Failed to process field '{key}': {e}")
            continue

    # Determine batch size from first tensor
    batch_size = 0
    if batch:
        first_tensor = next(iter(batch.values()))
        if not isinstance(first_tensor, torch.Tensor):
            raise TypeError(f"Expected tensor in batch, got {type(first_tensor)}")
        if first_tensor.dim() < 1:
            raise ValueError(f"Tensor must have at least 1 dimension, got shape {first_tensor.shape}")
        batch_size = first_tensor.shape[0]
    elif non_tensor_batch:
        # Estimate batch size from non-tensor data
        batch_size = _estimate_batch_size_from_non_tensor(non_tensor_batch)

    if batch_size == 0:
        logger.warning("Could not determine batch size, using default of 1")
        batch_size = 1

    # Create DataProto
    try:
        return DataProto(
            batch=TensorDict(batch, batch_size=batch_size),
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info.copy(),
        )
    except Exception as e:
        logger.error(f"Failed to create DataProto: {e}")
        raise


def _dataproto_to_tensordict(data: DataProto) -> TensorDict:
    """
    Convert DataProto to TensorDict for storage using NonTensorData.

    Args:
        data: DataProto to convert

    Returns:
        TensorDict containing the converted data

    Raises:
        ValueError: If data is invalid
        TypeError: If data types are unsupported
    """
    if not data:
        raise ValueError("data cannot be None or empty")

    # Start with tensor data
    tensor_dict = dict(data.batch)

    # Handle non-tensor data - convert to tensors for simplicity
    for key, value in data.non_tensor_batch.items():
        try:
            if isinstance(value, torch.Tensor):
                # Keep tensors as-is
                tensor_dict[key] = value
            elif isinstance(value, (list, tuple)) and len(value) == len(data):
                # Convert batch-aligned lists to tensors if possible
                if all(isinstance(item, (int, float)) for item in value):
                    tensor_dict[key] = torch.tensor(value, dtype=torch.float32)
                else:
                    # Skip non-numeric data
                    continue
            elif isinstance(value, (int, float, bool)):
                # Convert scalars to tensors
                tensor_dict[key] = torch.tensor([value] * len(data), dtype=torch.float32)
            else:
                # Skip complex types
                logger.debug(f"Skipping non-tensor field '{key}' with type {type(value)}")
                continue
        except Exception as e:
            logger.warning(f"Failed to convert non-tensor field '{key}': {e}")
            continue

    # Create TensorDict
    try:
        return TensorDict(**tensor_dict, batch_size=len(data))
    except Exception as e:
        logger.warning(f"TensorDict creation failed: {e}, trying fallback")
        # Fallback: create with batch_size parameter
        td = TensorDict({}, batch_size=len(data))
        for key, value in tensor_dict.items():
            try:
                td.set(key, value)
            except Exception as set_error:
                logger.warning(f"Failed to set field '{key}' in TensorDict: {set_error}")
        return td


def _handle_sync_with_async_client(
    batch_meta: BatchMeta,
    client: AsyncTransferQueueClient,
    other_args: tuple,
    other_kwargs: dict,
    async_wrapper_func: Callable,
) -> BatchMeta:
    """
    Handle synchronous function call with async client.

    Args:
        batch_meta: BatchMeta to process
        client: Async client for data operations
        other_args: Additional positional arguments
        other_kwargs: Additional keyword arguments
        async_wrapper_func: The async wrapper function to call

    Returns:
        Updated BatchMeta

    Raises:
        RuntimeError: If called in running event loop
    """
    # Check if we're in an event loop
    try:
        asyncio.get_running_loop()
        # We're in an event loop, this shouldn't happen for sync functions
        raise RuntimeError(
            "Cannot call synchronous decorated function with AsyncTransferQueueClient "
            "when an event loop is running. Use an async function instead."
        )
    except RuntimeError as e:
        if "no running event loop" in str(e):
            # No event loop, we can use asyncio.run
            # Reconstruct kwargs with client
            all_kwargs = other_kwargs.copy()
            all_kwargs["transfer_queue_client"] = client
            return asyncio.run(async_wrapper_func(batch_meta, *other_args, **all_kwargs))
        else:
            # Re-raise our specific error
            raise


def _estimate_batch_size_from_non_tensor(non_tensor_batch: dict[str, Any]) -> int:
    """
    Estimate batch size from non-tensor data.

    Args:
        non_tensor_batch: Dictionary of non-tensor data

    Returns:
        Estimated batch size, or 1 if cannot determine
    """
    for key, value in non_tensor_batch.items():
        if isinstance(value, (list, tuple)):
            return len(value)
    return 1


def dataproto_batchmeta_conversion_v2(
    func: Optional[Callable] = None, *, transfer_queue_client: Optional[AsyncTransferQueueClient] = None
) -> Callable:
    """
    Alternative decorator syntax that supports both @decorator and @decorator() usage.

    Args:
        func: Optional function to decorate
        transfer_queue_client: AsyncTransferQueueClient for data operations

    Returns:
        Decorated function or decorator

    Example:
        # Both syntaxes work:
        @dataproto_batchmeta_conversion_v2
        def my_func(data: DataProto) -> DataProto: ...

        @dataproto_batchmeta_conversion_v2(transfer_queue_client=client)
        def my_func(data: DataProto) -> DataProto: ...
    """

    def decorator(f: Callable) -> Callable:
        return dataproto_batchmeta_conversion(transfer_queue_client)(f)

    if func is not None:
        return decorator(func)
    return decorator
