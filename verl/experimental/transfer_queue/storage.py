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

import logging
import os
import time
from operator import itemgetter
from threading import Thread
from uuid import uuid4

import ray
import torch
import zmq
from ray.util import get_node_ip_address
from tensordict import NonTensorStack, TensorDict

from verl.experimental.transfer_queue.utils.utils import TransferQueueRole
from verl.experimental.transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    get_free_port,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

TQ_STORAGE_POLLER_TIMEOUT = os.environ.get("TQ_STORAGE_POLLER_TIMEOUT", 1000)
TQ_STORAGE_HANDSHAKE_TIMEOUT = int(os.environ.get("TQ_STORAGE_HANDSHAKE_TIMEOUT", 30))
TQ_DATA_UPDATE_RESPONSE_TIMEOUT = int(os.environ.get("TQ_DATA_UPDATE_RESPONSE_TIMEOUT", 600))


class StorageUnitData:
    """
    Class used for storing several elements, each element is composed of several fields and corresponding data, like:
    #####################################################
    # local_index | field_name1 | field_name2 | ...   #
    # 0           | item1       | item2       | ...   #
    # 1           | item3       | item4       | ...   #
    # 2           | item5       | item6       | ...   #
    #####################################################
    """

    def __init__(self, storage_size: int):
        # Dict containing field names and corresponding data in the field, e.g. {"field_name1": [data1, data2, ...]}
        self.field_data: dict[str, list] = {}

        # Maximum number of elements stored in storage unit
        self.storage_size = storage_size

    def get_data(self, fields: list[str], local_indexes: list[int]) -> TensorDict[str, list]:
        """
        Get data from storage unit according to given fields and local_indexes.

        param:
            fields: Field names used for getting data.
            local_indexes: Local indexes used for getting data.
        return:
            TensorDict with field names as keys, corresponding data list as values.
        """
        result: dict[str, list] = {}

        for field in fields:
            # Validate field name
            if field not in self.field_data:
                raise ValueError(
                    f"StorageUnitData get_data operation receive invalid field: {field} beyond {self.field_data.keys()}"
                )

            if len(local_indexes) == 1:
                # The unsqueeze op make the shape from n to (1, n)
                gathered_item = self.field_data[field][local_indexes[0]]
                if not isinstance(gathered_item, torch.Tensor):
                    result[field] = NonTensorStack(gathered_item).unsqueeze(0)
                else:
                    result[field] = gathered_item.unsqueeze(0)
            else:
                gathered_items = list(itemgetter(*local_indexes)(self.field_data[field]))

                if gathered_items:
                    all_tensors = all(isinstance(x, torch.Tensor) for x in gathered_items)
                    if all_tensors:
                        result[field] = torch.nested.as_nested_tensor(gathered_items)
                    else:
                        result[field] = NonTensorStack(*gathered_items)

        return TensorDict(result)

    def put_data(self, field_data: TensorDict[str, list], local_indexes: list[int]) -> None:
        """
        Put or update data into storage unit according to given field_data and local_indexes.

        param:
            field_data: Dict with field names as keys, corresponding data in the field as values.
            local_indexes: Local indexes used for putting data.
        """
        for f in field_data.keys():
            for i, idx in enumerate(local_indexes):
                # Validate local_indexes
                if idx < 0 or idx >= self.storage_size:
                    raise ValueError(
                        f"StorageUnitData put_data operation receive invalid local_index: {idx} beyond "
                        f"storage_size: {self.storage_size}"
                    )

                if f not in self.field_data:
                    # Initialize new field value list with None
                    self.field_data[f] = [None] * self.storage_size

                self.field_data[f][idx] = field_data[f][i]

    def clear(self, local_indexes: list[int]) -> None:
        """
        Clear data at specified local_indexes by setting all related fields to None.

        param:
            local_indexes: local_indexes to clear.
        """
        # Validate local_indexes
        for idx in local_indexes:
            if idx < 0 or idx >= self.storage_size:
                raise ValueError(
                    f"StorageUnitData clear operation receive invalid local_index: {idx} beyond "
                    f"storage_size: {self.storage_size}"
                )

        # Clear data at specified local_indexes
        for f in self.field_data:
            for idx in local_indexes:
                self.field_data[f][idx] = None


@ray.remote(num_cpus=1)
class TransferQueueStorageSimpleUnit:
    def __init__(self, storage_size: int):
        super().__init__()
        self.storage_unit_id = f"TQ_STORAGE_UNIT_{uuid4()}"
        self.storage_size = storage_size
        self.controller_infos: dict[str, ZMQServerInfo] = {}

        self.experience_data = StorageUnitData(self.storage_size)

        self.zmq_server_info = ZMQServerInfo.create(
            role=TransferQueueRole.STORAGE,
            id=str(self.storage_unit_id),
            ip=get_node_ip_address(),
            ports={"put_get_socket": get_free_port()},
        )
        self._init_zmq_socket()

    def _init_zmq_socket(self) -> None:
        """
        Initialize ZMQ socket connections between storage unit and controllers/clients:
        - controller_handshake_sockets:
            Handshake between storage unit and controllers.
        - data_status_update_sockets:
            Broadcast data update status from storage unit to controllers when handling put operation.
        - put_get_socket:
            Handle put/get requests from clients.
        """
        self.zmq_context = zmq.Context()

        self.controller_handshake_sockets: dict[str, zmq.Socket] = {}
        self.data_status_update_sockets: dict[str, zmq.Socket] = {}

        self.put_get_socket = create_zmq_socket(self.zmq_context, zmq.ROUTER)
        self.put_get_socket.bind(self.zmq_server_info.to_addr("put_get_socket"))

    def register_controller_info(self, controller_infos: dict[str, ZMQServerInfo]) -> None:
        """
        Build connections between storage unit and controllers, start put/get process.

        param:
            controller_infos: Dict with controller infos.
        """
        self.controller_infos = controller_infos

        self._init_zmq_sockets_with_controller_infos()
        self._connect_to_controller()
        self._start_process_put_get()

    def _init_zmq_sockets_with_controller_infos(self) -> None:
        """Initialize ZMQ sockets between storage unit and controllers for handshake."""
        for controller_id in self.controller_infos.keys():
            self.controller_handshake_sockets[controller_id] = create_zmq_socket(
                self.zmq_context,
                zmq.DEALER,
                identity=f"{self.storage_unit_id}-controller_handshake_sockets-{uuid4()}".encode(),
            )
            self.data_status_update_sockets[controller_id] = create_zmq_socket(
                self.zmq_context,
                zmq.DEALER,
                identity=f"{self.storage_unit_id}-data_status_update_sockets-{uuid4()}".encode(),
            )

    def _connect_to_controller(self) -> None:
        """Connect storage unit to all controllers."""
        connected_controllers: set[str] = set()

        # Create zmq poller for handshake confirmation between controller and storage unit
        poller = zmq.Poller()

        for controller_id, controller_info in self.controller_infos.items():
            self.controller_handshake_sockets[controller_id].connect(controller_info.to_addr("handshake_socket"))
            logger.debug(
                f"[{self.zmq_server_info.id}]: Handshake connection from storage unit id #{self.zmq_server_info.id} "
                f"to controller id #{controller_id} establish successfully."
            )

            # Send handshake request to controllers
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.HANDSHAKE,
                sender_id=self.zmq_server_info.id,
                body={
                    "storage_unit_id": self.storage_unit_id,
                    "storage_size": self.storage_size,
                },
            ).serialize()

            self.controller_handshake_sockets[controller_id].send(request_msg)
            logger.debug(
                f"[{self.zmq_server_info.id}]: Send handshake request from storage unit id #{self.zmq_server_info.id} "
                f"to controller id #{controller_id} successfully."
            )

            poller.register(self.controller_handshake_sockets[controller_id], zmq.POLLIN)

        start_time = time.time()
        while (
            len(connected_controllers) < len(self.controller_infos)
            and time.time() - start_time < TQ_STORAGE_HANDSHAKE_TIMEOUT
        ):
            socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT))

            for controller_handshake_socket in self.controller_handshake_sockets.values():
                if controller_handshake_socket in socks:
                    response_msg = ZMQMessage.deserialize(controller_handshake_socket.recv())

                    if response_msg.request_type == ZMQRequestType.HANDSHAKE_ACK:
                        connected_controllers.add(response_msg.sender_id)
                        logger.debug(
                            f"[{self.zmq_server_info.id}]: Get handshake ACK response from "
                            f"controller id #{str(response_msg.sender_id)} to storage unit id "
                            f"#{self.zmq_server_info.id} successfully."
                        )

        if len(connected_controllers) < len(self.controller_infos):
            logger.warning(
                f"[{self.zmq_server_info.id}]: Only get {len(connected_controllers)} / {len(self.controller_infos)} "
                f"successful handshake connections to controllers from storage unit id #{self.zmq_server_info.id}"
            )

    def _start_process_put_get(self) -> None:
        """Create a daemon thread and start put/get process."""
        self.process_put_get_thread = Thread(
            target=self._process_put_get, name=f"StorageUnitProcessPutGetThread-{self.zmq_server_info.id}", daemon=True
        )
        self.process_put_get_thread.start()

    def _process_put_get(self) -> None:
        """Process put_get_socket request."""
        poller = zmq.Poller()
        poller.register(self.put_get_socket, zmq.POLLIN)

        while True:
            socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT))

            if self.put_get_socket in socks:
                identity, serialized_msg = self.put_get_socket.recv_multipart()

                try:
                    request_msg = ZMQMessage.deserialize(serialized_msg)
                    operation = request_msg.request_type
                    logger.debug(f"[{self.zmq_server_info.id}]: receive operation: {operation}, message: {request_msg}")

                    if operation == ZMQRequestType.PUT_DATA:
                        response_msg = self._handle_put(request_msg)
                    elif operation == ZMQRequestType.GET_DATA:
                        response_msg = self._handle_get(request_msg)
                    elif operation == ZMQRequestType.CLEAR_DATA:
                        response_msg = self._handle_clear(request_msg)
                    else:
                        response_msg = ZMQMessage.create(
                            request_type=ZMQRequestType.PUT_GET_OPERATION_ERROR,
                            sender_id=self.zmq_server_info.id,
                            body={
                                "message": f"Storage unit id #{self.zmq_server_info.id} "
                                f"receive invalid operation: {operation}."
                            },
                        )
                except Exception as e:
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.PUT_GET_ERROR,
                        sender_id=self.zmq_server_info.id,
                        body={
                            "message": f"Storage unit id #{self.zmq_server_info.id} occur error in processing "
                            f"put/get/clear request, detail error message: {str(e)}."
                        },
                    )

                self.put_get_socket.send_multipart([identity, response_msg.serialize()])

    def _handle_put(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle put request, add or update data into storage unit.

        param:
            data_parts: ZMQMessage from client.
        return:
            Put data success response ZMQMessage.
        """
        try:
            global_indexes = data_parts.body["global_indexes"]
            local_indexes = data_parts.body["local_indexes"]
            field_data = data_parts.body["field_data"]  # field_data should be in {field_name: [real data]} format.

            self.experience_data.put_data(field_data, local_indexes)

            # After put operation finish, send a message to the client
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_RESPONSE, sender_id=self.zmq_server_info.id, body={}
            )

            # Gather per-tensor dtype and shape information for each field
            # global_indexes, local_indexes, and field_data correspond one-to-one
            per_tensor_dtypes: dict[int, torch.dtype] = {}
            per_tensor_shapes: dict[int, torch.Size] = {}

            # Initialize the data structure for each global index
            for global_idx in global_indexes:
                per_tensor_dtypes[global_idx] = {}
                per_tensor_shapes[global_idx] = {}

            # For each field, extract dtype and shape for each sample
            for field in field_data.keys():
                for i, data_item in enumerate(field_data[field]):
                    global_idx = global_indexes[i]
                    per_tensor_dtypes[global_idx][field] = data_item.dtype if hasattr(data_item, "dtype") else None
                    per_tensor_shapes[global_idx][field] = data_item.shape if hasattr(data_item, "shape") else None

            # Broadcast data update message to all controllers with per-tensor dtype/shape information
            self._notify_data_update(list(field_data.keys()), global_indexes, per_tensor_dtypes, per_tensor_shapes)
            return response_msg
        except Exception as e:
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_ERROR,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": f"Failed to put data into storage unit id "
                    f"#{self.zmq_server_info.id}, detail error message: {str(e)}"
                },
            )

    def _notify_data_update(self, fields, global_indexes, dtypes, shapes) -> None:
        """
        Broadcast data status update to all controllers.

        param:
            fields: data update related fields.
            global_indexes: data update related global_indexes.
            dtypes: per-tensor dtypes for each field, in {global_index: {field: dtype}} format.
            shapes: per-tensor shapes for each field, in {global_index: {field: shape}} format.
        """
        # Create zmq poller for notifying data update information
        poller = zmq.Poller()

        # Connect data status update socket to all controllers
        for controller_id, controller_info in self.controller_infos.items():
            data_status_update_socket = self.data_status_update_sockets[controller_id]
            data_status_update_socket.connect(controller_info.to_addr("data_status_update_socket"))
            logger.debug(
                f"[{self.zmq_server_info.id}]: Data status update connection from "
                f"storage unit id #{self.zmq_server_info.id} to "
                f"controller id #{controller_id} establish successfully."
            )

            try:
                poller.register(data_status_update_socket, zmq.POLLIN)

                request_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE,
                    sender_id=self.zmq_server_info.id,
                    body={
                        "fields": fields,
                        "global_indexes": global_indexes,
                        "dtypes": dtypes,
                        "shapes": shapes,
                    },
                ).serialize()

                data_status_update_socket.send(request_msg)
                logger.debug(
                    f"[{self.zmq_server_info.id}]: Send data status update request "
                    f"from storage unit id #{self.zmq_server_info.id} "
                    f"to controller id #{controller_id} successfully."
                )
            except Exception as e:
                request_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ERROR,
                    sender_id=self.zmq_server_info.id,
                    body={
                        "message": f"Failed to notify data status update information from "
                        f"storage unit id #{self.zmq_server_info.id}, "
                        f"detail error message: {str(e)}"
                    },
                ).serialize()

                data_status_update_socket.send(request_msg)

        # Make sure all controllers successfully receive data status update information.
        response_controllers: set[str] = set()
        start_time = time.time()

        while (
            len(response_controllers) < len(self.controller_infos)
            and time.time() - start_time < TQ_DATA_UPDATE_RESPONSE_TIMEOUT
        ):
            socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT))

            for data_status_update_socket in self.data_status_update_sockets.values():
                if data_status_update_socket in socks:
                    response_msg = ZMQMessage.deserialize(data_status_update_socket.recv())

                    if response_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ACK:
                        response_controllers.add(response_msg.sender_id)
                        logger.debug(
                            f"[{self.zmq_server_info.id}]: Get data status update ACK response "
                            f"from controller id #{response_msg.sender_id} "
                            f"to storage unit id #{self.zmq_server_info.id} successfully."
                        )

        if len(response_controllers) < len(self.controller_infos):
            logger.warning(
                f"[{self.zmq_server_info.id}]: Storage unit id #{self.zmq_server_info.id} "
                f"only get {len(response_controllers)} / {len(self.controller_infos)} "
                f"data status update ACK responses from controllers."
            )

    def _handle_get(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle get request, return data from storage unit.

        param:
            data_parts: ZMQMessage from client.
        return:
            Get data success response ZMQMessage, containing target data.
        """
        try:
            fields = data_parts.body["fields"]
            local_indexes = data_parts.body["local_indexes"]

            result_data = self.experience_data.get_data(fields, local_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_RESPONSE,
                sender_id=self.zmq_server_info.id,
                body={
                    "data": result_data,
                },
            )
        except Exception as e:
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_ERROR,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": f"Failed to get data from storage unit id #{self.zmq_server_info.id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    def _handle_clear(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle clear request, clear data in storage unit according to given local_indexes.

        param:
            data_parts: ZMQMessage from client, including target local_indexes.
        return:
            Clear data success response ZMQMessage.
        """
        try:
            local_indexes = data_parts.body["local_indexes"]

            self.experience_data.clear(local_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_RESPONSE,
                sender_id=self.zmq_server_info.id,
                body={"message": f"Clear data in storage unit id #{self.zmq_server_info.id} successfully."},
            )
        except Exception as e:
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_ERROR,
                sender_id=self.zmq_server_info.id,
                body={
                    "message": f"Failed to clear data in storage unit id #{self.zmq_server_info.id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    def get_zmq_server_info(self) -> ZMQServerInfo:
        return self.zmq_server_info
