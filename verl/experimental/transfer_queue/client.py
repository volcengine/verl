# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Huawei Ltd. and/or its affiliates
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
import logging
import os
from functools import wraps
from typing import Any, Callable, Optional, Union
from uuid import uuid4

import ray
import torch
import zmq
import zmq.asyncio
from tensordict import NonTensorStack, TensorDict

from verl.experimental.transfer_queue.controller import TransferQueueController
from verl.experimental.transfer_queue.metadata import (
    BatchMeta,
    StorageMetaGroup,
)
from verl.experimental.transfer_queue.storage import TransferQueueStorageSimpleUnit
from verl.experimental.transfer_queue.utils.utils import (
    TransferQueueRole,
)
from verl.experimental.transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncTransferQueueClient:
    def __init__(
        self,
        client_id: str,
        controller_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
        storage_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
    ):
        self.client_id = client_id

        self._controllers: dict[str, ZMQServerInfo] = {}
        self._storages: dict[str, ZMQServerInfo] = {}
        self._register_servers(TransferQueueRole.CONTROLLER, controller_infos)
        self._register_servers(TransferQueueRole.STORAGE, storage_infos)

    def _register_servers(
        self,
        role: TransferQueueRole,
        server_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
    ):
        mapping = self._controllers if role == TransferQueueRole.CONTROLLER else self._storages

        if not isinstance(server_infos, dict):
            server_infos = {server_infos.id: server_infos}

        for info in server_infos.values():
            if not isinstance(info, ZMQServerInfo):
                raise ValueError(f"Invalid server info for {role} {info.id}")

            if info.id not in mapping:
                mapping[info.id] = info
                logger.info(f"[{self.client_id}]: Registered {role} server {info.id} at {info.ip}")
            else:
                logger.warning(f"[{self.client_id}]: Server {info.id} already registered, skipping")

    @staticmethod
    def dynamic_socket(target_role: TransferQueueRole, socket_name: str):
        """Decorator to auto-manage ZMQ sockets for Controller/Storage servers (create -> connect -> inject -> close).

        Args:
            target_role (TransferQueueRole): Server type to connect to. Must be one of:
                - `TransferQueueRole.CONTROLLER`
                - `TransferQueueRole.STORAGE`
            socket_name (str): Port name (from server config) to use for ZMQ connection (e.g., "data_req_port").

        Decorated Function Rules:
            1. Must be an async class method (needs `self`).
            2. `self` requires:
            - `_controllers`/`_storages`: Server registries (match `target_role`).
            - `client_id`: Unique client ID (for socket identity).
            3. Specify target server via:
            - `target_controller` (for Controller) or `target_storage` (for Storage) arg.
            - Controller role: Uses first registered server if no ID is given.
            4. Receives ZMQ socket via `socket` keyword arg (injected by decorator).
        """

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                if target_role == TransferQueueRole.CONTROLLER:
                    servers = self._controllers
                    target = "target_controller"
                elif target_role == TransferQueueRole.STORAGE:
                    servers = self._storages
                    target = "target_storage"
                else:
                    raise ValueError("Invalid target_role, must be CONTROLLER or STORAGE")

                server_key = kwargs.get(target)
                if server_key is None:
                    for arg in args:
                        if isinstance(arg, str) and arg in servers.keys():
                            server_key = arg
                            break
                if server_key is None and target == "target_controller":
                    server_key = next(iter(servers.keys()))

                server_info = servers.get(server_key)
                if not server_info:
                    raise RuntimeError(f"Server {server_key} not found in registered {target_role} servers")

                context = zmq.asyncio.Context()
                address = f"tcp://{server_info.ip}:{server_info.ports.get(socket_name)}"
                identity = f"{self.client_id}_to_{server_info.id}_{uuid4()}".encode()
                sock = create_zmq_socket(context, zmq.DEALER, identity=identity)

                try:
                    sock.connect(address)
                    logger.info(
                        f"[{self.client_id}]: Connected to {target_role} {server_info.id} at {address} "
                        f"with identity {identity.decode()}"
                    )

                    kwargs["socket"] = sock
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    logger.error(
                        f"[{self.client_id}]: Error in socket operation with {target_role} {server_info.id}: {e}"
                    )
                    raise
                finally:
                    try:
                        if not sock.closed:
                            sock.setsockopt(zmq.LINGER, -1)
                            sock.close()
                        sock.close(linger=0)
                    except Exception as e:
                        logger.warning(
                            f"[{self.client_id}]: Error closing socket to {target_role} {server_info.id}: {e}"
                        )

                    context.term()

            return wrapper

        return decorator

    @dynamic_socket(target_role=TransferQueueRole.CONTROLLER, socket_name="request_handle_socket")
    async def async_get_meta(
        self,
        data_fields: list[str],
        batch_size: int,
        global_step: int,
        mode: str = "fetch",
        get_n_samples: bool = False,
        task_name: Optional[str] = None,
        target_controller: Optional[str] = None,
        socket: Optional[zmq.asyncio.Socket] = None,
    ) -> BatchMeta:
        """Asynchronously fetches data metadata via ZMQ from the target controller.

        Args:
            data_fields (list[str]): List of fields to retrieve metadata for
            batch_size (int): Processing batch size
            global_step (int): Current training/processing step
            mode (str): Data fetch mode (TODO(hz): more details to be added)
            get_n_samples (bool): TODO(hz): more details to be added
            task_name (str): Optional task name associated with the request
            target_controller (str): ID of the target controller to send the request to
            socket (zmq.asyncio.Socket): ZMQ async socket for message transmission

        Returns:
            BatchMeta: Metadata object containing data structure, sample info, etc.
        """
        assert socket is not None
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_META,
            sender_id=self.client_id,
            receiver_id=target_controller,
            body={
                "data_fields": data_fields,
                "batch_size": batch_size,
                "global_step": global_step,
                "mode": mode,
                "get_n_samples": get_n_samples,
                "task_name": task_name,
            },
        )

        try:
            await socket.send(request_msg.serialize())
            response = await socket.recv()
            response_msg = ZMQMessage.deserialize(response)
            logger.debug(
                f"[{self.client_id}]: Client get datameta response: {response_msg} from controller {target_controller}"
            )

            if response_msg.request_type == ZMQRequestType.GET_META_RESPONSE:
                metadata = response_msg.body["metadata"]
                return metadata
            else:
                raise RuntimeError(
                    f"[{self.client_id}]: Failed to get metadata from controller {target_controller}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"[{self.client_id}]: Error in get_meta: {str(e)}") from e

    async def async_put(
        self,
        data: TensorDict,
        metadata: Optional[BatchMeta] = None,
        global_step: Optional[int] = None,
    ):
        """Asynchronously writes data to appropriate Storage Units based on metadata.

        If metadata isn't provided, it will be created automatically using the insert mode
        with the provided data_columns and global_step.

        Args:
            data (torch.Tensor | tensordict.TensorDict): Data to write, either a Tensor or TensorDict
            metadata (BatchMeta, optional): Optional metadata containing index and storage unit information
            global_step (int, optional): Current step (required if no metadata is provided)

        """
        if metadata is None:
            assert global_step is not None, "global_steps must be provided if metadata is not given"

            metadata = await self.async_get_meta(
                data_fields=list(data.keys()),
                batch_size=data.batch_size[0],
                global_step=global_step,
                mode="insert",
            )

        if not metadata or metadata.size == 0:
            raise ValueError("metadata cannot be none or empty")
        logger.debug(f"[{self.client_id}]: Put data with data: {data}")
        tasks = [
            self._put_to_storage(get_transfer_info(meta_group, data), target_storage=storage_id)
            for storage_id, meta_group in metadata.storage_meta_groups.items()
        ]
        await asyncio.gather(*tasks)

        logger.info(
            f"[{self.client_id}]: step {global_step} put {metadata.size} samples to storage units successfully."
        )

    @dynamic_socket(target_role=TransferQueueRole.STORAGE, socket_name="put_get_socket")
    async def _put_to_storage(self, storage_unit_data, target_storage=None, socket=None):
        """
        Send data to a specific storage unit.
        """
        global_indexes = storage_unit_data["global_indexes"]
        local_indexes = storage_unit_data["local_indexes"]
        field_data = TensorDict(
            {
                field: (
                    torch.nested.as_nested_tensor(storage_unit_data["field_data"][field])
                    if storage_unit_data["field_data"][field]
                    and all(isinstance(x, torch.Tensor) for x in storage_unit_data["field_data"][field])
                    else NonTensorStack(*storage_unit_data["field_data"][field])
                )
                for field in storage_unit_data["field_data"]
            }
        )

        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id=self.client_id,
            receiver_id=target_storage,
            body={"global_indexes": global_indexes, "local_indexes": local_indexes, "field_data": field_data},
        )
        try:
            await socket.send(request_msg.serialize())
            serialized = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized)

            if response_msg.request_type != ZMQRequestType.PUT_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to put data to storage unit {target_storage}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"Error in put to storage unit {target_storage}: {str(e)}") from e

    @dynamic_socket(target_role=TransferQueueRole.STORAGE, socket_name="put_get_socket")
    async def _get_from_storage(self, index_data, target_storage=None, socket=None):
        global_indexes = index_data["global_indexes"]
        local_indexes = index_data["local_indexes"]
        fields = index_data["fields"]

        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,
            sender_id=self.client_id,
            receiver_id=target_storage,
            body={"local_indexes": local_indexes, "fields": fields},
        )

        try:
            await socket.send(request_msg.serialize())
            serialized = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized)
            logger.info(f"[{self.client_id}]: get data response from storage unit {target_storage}: {response_msg}")

            if response_msg.request_type == ZMQRequestType.GET_DATA_RESPONSE:
                # Return data and index information from this storage unit
                storage_unit_data = response_msg.body["data"]
                return global_indexes, fields, storage_unit_data
            else:
                raise RuntimeError(
                    f"Failed to get data from storage unit {target_storage}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"Error getting data from storage unit {target_storage}: {str(e)}") from e

    async def async_get_data(self, metadata: BatchMeta) -> TensorDict:
        """Asynchronously fetches data via Storage Units and organizes it into a TensorDict.

        Args:
            metadata (BatchMeta): Object containing:
                - Data location info (which Storage Units hold the data)
                - `global_indexes` to determine the ordering of merged results

        Returns:
            tensordict.TensorDict with:
                - Requested data fields (e.g., "prompt_token_ids", "response_token_ids").
                - "global_indexes" key: Maps each sample to its original global index.

        Example:
            >>> returned_td = await async_get_data(metadata)
            >>> returned_td.keys()
            dict_keys(['prompt_token_ids', 'response_token_ids', 'global_indexes'])
            >>> returned_td["prompt_token_ids"].shape  # Batch size 4, seq length 128
            torch.Size([4, 128])
            >>> returned_td["global_indexes"]  # Preserves original global order
            tensor([7, 4, 6, 5])

        Note:
            Why track `global_indexes`?
            - Batches may be rearranged during task processing. `global_indexes` retains the original
            mapping to Storage Units, enabling correct data writing back to Storage Units later.

        """
        if not metadata or metadata.size == 0:
            return TensorDict({}, batch_size=0)

        # Use optimized retrieval with direct storage group access
        tasks = [
            self._get_from_storage(meta_group.get_transfer_info(), target_storage=storage_id)
            for storage_id, meta_group in metadata.storage_meta_groups.items()
        ]

        results = await asyncio.gather(*tasks)

        # global_index: {field1: value, field2: value, ...}
        storage_data: dict[int, dict[str, torch.Tensor]] = {}
        for global_indexes, fields, storage_unit_data in results:
            for idx, global_idx in enumerate(global_indexes):
                if global_idx not in storage_data:
                    storage_data[global_idx] = {}
                for field in fields:
                    storage_data[global_idx][field] = storage_unit_data[field][idx]

        ordered_data: dict[str, torch.Tensor] = {field: [] for field in metadata.fields}
        for global_idx in metadata.global_indexes:
            for field in metadata.fields:
                ordered_data[field].append(storage_data[global_idx][field])

        tensor_data = {
            field: (
                torch.stack(torch.nested.as_nested_tensor(v).unbind())
                if v
                and all(isinstance(item, torch.Tensor) for item in v)
                and all(item.shape == v[0].shape for item in v)
                else (
                    torch.nested.as_nested_tensor(v)
                    if v and all(isinstance(item, torch.Tensor) for item in v)
                    else NonTensorStack(*v)
                )
            )
            for field, v in ordered_data.items()
        }
        tensor_data["global_indexes"] = torch.tensor(metadata.global_indexes)

        return TensorDict(tensor_data, batch_size=len(storage_data))

    async def async_clear(self, global_step: int):
        """Asynchronously clears data from all storage units and controller metadata.

        Args:
            global_step (int): The training step associated with the clear operation
        """
        try:
            target_controller = next(iter(self._controllers.keys()))
            metadata = await self._get_clear_meta(global_step, target_controller)

            tasks = []

            for target_controller in self._controllers.keys():
                tasks.append(self._clear_controller(global_step, target_controller))

            # Group samples by storage unit for clearing
            for target_storage, group in metadata.storage_meta_groups.items():
                group_info = group.get_transfer_info()
                if target_storage not in self._storages:
                    logger.warning(
                        f"[{self.client_id}]: Storage unit {target_storage} not registered, skipping clear operation."
                    )
                    continue
                tasks.append(
                    self._clear_storage_unit(
                        group_info["local_indexes"],
                        target_storage,
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"[{self.client_id}]: Error in clear operation task {i}: {result}")

            logger.info(f"[{self.client_id}]: Clear operation for global_step {global_step} completed.")
        except Exception as e:
            raise RuntimeError(f"Error in clear operation: {str(e)}") from e

    @dynamic_socket(target_role=TransferQueueRole.CONTROLLER, socket_name="request_handle_socket")
    async def _get_clear_meta(self, global_step: int, target_controller=None, socket=None):
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_CLEAR_META,
            sender_id=self.client_id,
            receiver_id=target_controller,
            body={"global_step": global_step},
        )

        await socket.send(request_msg.serialize())
        serialized = await socket.recv()
        response_msg = ZMQMessage.deserialize(serialized)

        if response_msg.request_type != ZMQRequestType.GET_CLEAR_META_RESPONSE:
            raise RuntimeError(
                f"Failed to get metadata for clear operation: {response_msg.body.get('message', 'Unknown error')}"
            )

        return response_msg.body["metadata"]

    @dynamic_socket(target_role=TransferQueueRole.CONTROLLER, socket_name="request_handle_socket")
    async def _clear_controller(self, global_step, target_controller=None, socket=None):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_META,
                sender_id=self.client_id,
                receiver_id=target_controller,
                body={"global_step": global_step},
            )

            await socket.send(request_msg.serialize())
            serialized_msg = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized_msg)

            if response_msg.request_type != ZMQRequestType.CLEAR_META_RESPONSE:
                raise RuntimeError(
                    f"Failed to clear controller {target_controller}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )

            logger.info(
                f"[{self.client_id}]: Successfully clear controller {target_controller} for global_step {global_step}"
            )
        except Exception as e:
            logger.error(f"[{self.client_id}]: Error clearing controller {target_controller}: {str(e)}")
            raise

    @dynamic_socket(target_role=TransferQueueRole.STORAGE, socket_name="put_get_socket")
    async def _clear_storage_unit(self, local_indexes, target_storage=None, socket=None):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA,
                sender_id=self.client_id,
                receiver_id=target_storage,
                body={"local_indexes": local_indexes},
            )

            await socket.send(request_msg.serialize())
            serialized_msg = await socket.recv()
            response_msg = ZMQMessage.deserialize(serialized_msg)

            if response_msg.request_type != ZMQRequestType.CLEAR_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to clear storage {target_storage}: {response_msg.body.get('message', 'Unknown error')}"
                )

            logger.info(f"[{self.client_id}]: Successfully clear storage unit {target_storage}")
        except Exception as e:
            logger.error(f"[{self.client_id}]: Error clearing storage unit {target_storage}: {str(e)}")
            raise


class TransferQueueClient(AsyncTransferQueueClient):
    def __init__(
        self,
        client_id: str,
        controller_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
        storage_infos: ZMQServerInfo | dict[Any, ZMQServerInfo],
    ):
        super().__init__(
            client_id,
            controller_infos,
            storage_infos,
        )

    def put(self, data: TensorDict, metadata: Optional[BatchMeta] = None, global_step: Optional[int] = None):
        return asyncio.run(self.async_put(data, metadata, global_step))

    def get_meta(
        self,
        data_fields: list[str],
        batch_size: int,
        global_step: int,
        get_n_samples: bool = False,
        task_name: Optional[str] = None,
    ) -> BatchMeta:
        return asyncio.run(
            self.async_get_meta(
                data_fields=data_fields,
                batch_size=batch_size,
                global_step=global_step,
                get_n_samples=get_n_samples,
                task_name=task_name,
            )
        )

    def get_data(self, metadata: BatchMeta) -> TensorDict:
        return asyncio.run(self.async_get_data(metadata))

    def clear(self, global_step: int):
        return asyncio.run(self.async_clear(global_step))


def _add_field_data(
    transfer_dict: dict[str, Any], storage_meta_group: StorageMetaGroup, data: TensorDict
) -> dict[str, Any]:
    """Helper function to add field data to the transfer dictionary"""
    field_names = transfer_dict["fields"]
    for fname in field_names:
        if fname in data.keys():
            transfer_dict["field_data"][fname] = []
            for sample_meta in storage_meta_group.sample_metas:
                transfer_dict["field_data"][fname].append(data[fname][sample_meta.batch_index])
    return transfer_dict


def get_transfer_info(
    storage_meta_group: StorageMetaGroup,
    data: TensorDict,
) -> dict[str, Any]:
    """Convert to dictionary format with field data for put operations"""
    result = storage_meta_group.get_transfer_info(field_names=data.keys())
    result = _add_field_data(result, storage_meta_group, data)
    return result


def process_zmq_server_info(handlers: dict[Any, Union[TransferQueueController, TransferQueueStorageSimpleUnit]]):  # noqa: UP007
    server_info = {}
    for name, handler in handlers.items():
        server_info[name] = ray.get(handler.get_zmq_server_info.remote())  # type: ignore[attr-defined]
    return server_info
