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
from functools import wraps
from typing import Any

import numpy as np
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.transfer_queue import (
    AsyncTransferQueueClient,
    BatchMeta,
    ZMQServerInfo,
)
from verl.protocol import DataProto

_TRANSFER_QUEUE_CLIENT = None


def create_transferqueue_client(
    client_id: str,
    controller_infos: dict[Any, ZMQServerInfo],
    storage_infos: dict[Any, ZMQServerInfo],
) -> None:
    global _TRANSFER_QUEUE_CLIENT
    _TRANSFER_QUEUE_CLIENT = AsyncTransferQueueClient(client_id, controller_infos, storage_infos)


def get_transferqueue_client() -> AsyncTransferQueueClient:
    return _TRANSFER_QUEUE_CLIENT


def _find_batchmeta(*args, **kwargs):
    for arg in args:
        if isinstance(arg, BatchMeta):
            return arg
    for v in kwargs.values():
        if isinstance(v, BatchMeta):
            return v
    return None


def _tensordict_to_dataproto(tensordict: TensorDict):
    batch = {}
    non_tensor_batch = {}
    batch_size = None
    for k, v in tensordict.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v
            if batch_size is None:
                batch_size = v.shape[:1]
        elif isinstance(v, NonTensorStack):
            non_tensor_batch[k] = np.array([elem.data for elem in v], dtype=object)
        else:
            non_tensor_batch[k] = v
    if len(batch) == 0 and len(non_tensor_batch) == 0:
        batch_size = (0,)
    return DataProto(
        batch=TensorDict(batch, batch_size=batch_size),
        non_tensor_batch=non_tensor_batch,
    )


def _batchmeta_to_dataproto(batchmeta: BatchMeta):
    tensordict = asyncio.run(_TRANSFER_QUEUE_CLIENT.async_get_data(batchmeta))
    result_dataproto = _tensordict_to_dataproto(tensordict)
    result_dataproto.meta_info = batchmeta.extra_info.copy()

    return result_dataproto


async def _async_batchmeta_to_dataproto(batchmeta: BatchMeta):
    tensordict = await _TRANSFER_QUEUE_CLIENT.async_get_data(batchmeta)
    result_dataproto = _tensordict_to_dataproto(tensordict)
    result_dataproto.meta_info = batchmeta.extra_info.copy()

    return result_dataproto


def _dataproto_to_tensordict(data: DataProto):
    result_dict = {}

    if data.batch is not None:
        result_dict.update(data.batch)
    
    batch_size = (0,)
    if data.batch is not None:
        batch_size = data.batch.batch_size
    elif data.non_tensor_batch is not None and len(data.non_tensor_batch) > 0:
        batch_size = (len(next(iter(data.non_tensor_batch.values()))),)

    if data.non_tensor_batch is not None:
        for k, v in data.non_tensor_batch.items():
            result_dict[k] = NonTensorData(data=v, batch_size=batch_size)

    if data.meta_info == {} or data.meta_info is None:
        result_dict["meta_info"] = NonTensorData(data=[None] * batch_size[0], batch_size=batch_size)
    else:
        result_dict["meta_info"] = NonTensorData(data=[data.meta_info] * batch_size[0], batch_size=batch_size)
    return TensorDict(result_dict, batch_size=batch_size)


def _update_batchmeta_with_output(output: DataProto, batchmeta: BatchMeta):
    tensordict = _dataproto_to_tensordict(output)
    if len(output) > 0:
        batchmeta.add_fields(tensordict)
        asyncio.run(_TRANSFER_QUEUE_CLIENT.async_put(data=tensordict, metadata=batchmeta))

    for k, v in output.meta_info.items():
        batchmeta.set_extra_info(k, v)


async def _async_update_batchmeta_with_output(output, batchmeta: BatchMeta):
    tensordict = _dataproto_to_tensordict(output)
    batchmeta.add_fields(tensordict)
    await _TRANSFER_QUEUE_CLIENT.async_put(data=tensordict, metadata=batchmeta)

    for k, v in output.meta_info.items():
        batchmeta.set_extra_info(k, v)


def batchmeta_dataproto_pipe(put_data: bool = True):
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

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        return wrapper

    return decorator
