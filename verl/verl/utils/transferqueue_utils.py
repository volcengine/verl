# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import inspect
import os
import threading
from functools import wraps
from typing import Any, Callable

from tensordict import TensorDict

try:
    from transfer_queue import (
        AsyncTransferQueueClient,
        BatchMeta,
        ZMQServerInfo,
    )

except ImportError:
    # TODO: Use a hacky workaround for ImportError since
    # transfer_queue isn't a default verl dependency.
    class BatchMeta:
        pass


from verl.protocol import DataProto

_TRANSFER_QUEUE_CLIENT = None
_VAL_TRANSFER_QUEUE_CLIENT = None

is_transferqueue_enabled = os.environ.get("TRANSFER_QUEUE_ENABLE", False)


def create_transferqueue_client(
    client_id: str,
    controller_infos: dict[Any, "ZMQServerInfo"],
    storage_infos: dict[Any, "ZMQServerInfo"],
) -> None:
    global _TRANSFER_QUEUE_CLIENT
    global _VAL_TRANSFER_QUEUE_CLIENT
    if "val" in client_id:
        _VAL_TRANSFER_QUEUE_CLIENT = AsyncTransferQueueClient(client_id, controller_infos, storage_infos)
    else:
        _TRANSFER_QUEUE_CLIENT = AsyncTransferQueueClient(client_id, controller_infos, storage_infos)


def get_transferqueue_client() -> "AsyncTransferQueueClient":
    return _TRANSFER_QUEUE_CLIENT


def get_val_transferqueue_client() -> "AsyncTransferQueueClient":
    return _VAL_TRANSFER_QUEUE_CLIENT


def _run_async_in_temp_loop(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    # Use a temporary event loop in a new thread because event
    # loop may already exist in server mode
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=tmp_event_loop.run_forever,
        name="batchmeta dataproto converter",
        daemon=True,
    )

    def run_coroutine(coroutine):
        if not thread.is_alive():
            thread.start()
        future = asyncio.run_coroutine_threadsafe(coroutine, tmp_event_loop)
        return future.result()

    async def stop_loop():
        tmp_event_loop.stop()

    try:
        return run_coroutine(async_func(*args, **kwargs))
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
            thread.join()


def _find_batchmeta(*args, **kwargs):
    for arg in args:
        if isinstance(arg, BatchMeta):
            return arg
    for v in kwargs.values():
        if isinstance(v, BatchMeta):
            return v
    return None


async def _async_batchmeta_to_dataproto(batchmeta: "BatchMeta") -> DataProto:
    if batchmeta.samples == [] or batchmeta.samples is None:
        return DataProto(
            batch=TensorDict({}, batch_size=(0,)),
            non_tensor_batch={},
            meta_info=batchmeta.extra_info.copy(),
        )

    if batchmeta.extra_info.get("validate", False):
        tensordict = await _VAL_TRANSFER_QUEUE_CLIENT.async_get_data(batchmeta)
    else:
        tensordict = await _TRANSFER_QUEUE_CLIENT.async_get_data(batchmeta)
    return DataProto.from_tensordict(tensordict, meta_info=batchmeta.extra_info.copy())


def _batchmeta_to_dataproto(batchmeta: "BatchMeta") -> DataProto:
    return _run_async_in_temp_loop(_async_batchmeta_to_dataproto, batchmeta)


async def _async_update_batchmeta_with_output(output: DataProto, batchmeta: "BatchMeta") -> None:
    for k, v in output.meta_info.items():
        batchmeta.set_extra_info(k, v)

    if len(output) > 0:
        tensordict = output.to_tensordict()
        # pop meta_info
        for key in output.meta_info.keys():
            tensordict.pop(key)
        batchmeta.add_fields(tensordict)
        if batchmeta.extra_info.get("validate", False):
            await _VAL_TRANSFER_QUEUE_CLIENT.async_put(data=tensordict, metadata=batchmeta)
        else:
            await _TRANSFER_QUEUE_CLIENT.async_put(data=tensordict, metadata=batchmeta)


def _update_batchmeta_with_output(output: DataProto, batchmeta: "BatchMeta") -> None:
    _run_async_in_temp_loop(_async_update_batchmeta_with_output, output, batchmeta)


def tqbridge(put_data: bool = True):
    """ "Creates a decorator for bridging BatchMeta and DataProto.

    This decorator automatically handles conversions between `BatchMeta` and
    `DataProto` in function parameters, and decides whether to sync function
    output back to `BatchMeta` based on configuration(`put_data`). It supports
    both synchronous and asynchronous functions (async def), and can control
    whether to enable enhanced logic via the global `HAS_TQ` variable (when disabled,
    simply calls the original function as-is).

    Args:
        put_data: Whether put the DataProto into Storage after func return.
                  If True, after function execution, the output result will be
                  updated to `BatchMeta` and `BatchMeta` will be returned;
                  If False, the function output result will be returned directly.
                  Defaults to True.

    Returns:
        A decorator function used to decorate target functions (synchronous or asynchronous).
    """

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            batchmeta = _find_batchmeta(*args, **kwargs)
            if batchmeta is None:
                return func(*args, **kwargs)
            else:
                args = [_batchmeta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
                kwargs = {k: _batchmeta_to_dataproto(v) if isinstance(v, BatchMeta) else v for k, v in kwargs.items()}
                output = func(*args, **kwargs)
                if put_data:
                    _update_batchmeta_with_output(output, batchmeta)
                    return batchmeta
                else:
                    return output

        @wraps(func)
        async def async_inner(*args, **kwargs):
            batchmeta = _find_batchmeta(*args, **kwargs)
            if batchmeta is None:
                return await func(*args, **kwargs)
            else:
                args = [await _async_batchmeta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
                kwargs = {
                    k: await _async_batchmeta_to_dataproto(v) if isinstance(v, BatchMeta) else v
                    for k, v in kwargs.items()
                }
                output = await func(*args, **kwargs)
                if put_data:
                    await _async_update_batchmeta_with_output(output, batchmeta)
                    return batchmeta
                return output

        @wraps(func)
        def dummy_inner(*args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        async def dummy_async_inner(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper_inner = inner if is_transferqueue_enabled else dummy_inner
        wrapper_async_inner = async_inner if is_transferqueue_enabled else dummy_async_inner

        wrapper = wrapper_async_inner if inspect.iscoroutinefunction(func) else wrapper_inner
        return wrapper

    return decorator
