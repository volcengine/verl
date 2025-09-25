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

import pickle
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import psutil
import zmq
from typing_extensions import Self

from verl.experimental.transfer_queue.utils.utils import (
    ExplicitEnum,
    TransferQueueRole,
)


class ZMQRequestType(ExplicitEnum):
    # HANDSHAKE
    HANDSHAKE = "HANDSHAKE"  # TransferQueueStorageUnit -> TransferQueueController
    HANDSHAKE_ACK = "HANDSHAKE_ACK"  # TransferQueueController  -> TransferQueueStorageUnit

    # DATA_OPERATION
    GET_DATA = "GET"
    PUT_DATA = "PUT"
    GET_DATA_RESPONSE = "GET_DATA_RESPONSE"
    PUT_DATA_RESPONSE = "PUT_DATA_RESPONSE"
    CLEAR_DATA = "CLEAR_DATA"
    CLEAR_DATA_RESPONSE = "CLEAR_DATA_RESPONSE"

    PUT_GET_OPERATION_ERROR = "PUT_GET_OPERATION_ERROR"
    PUT_GET_ERROR = "PUT_GET_ERROR"
    PUT_ERROR = "PUT_ERROR"
    GET_ERROR = "GET_ERROR"
    CLEAR_DATA_ERROR = "CLEAR_DATA_ERROR"

    # META_OPERATION
    GET_META = "GET_META"
    GET_META_RESPONSE = "GET_META_RESPONSE"
    GET_CLEAR_META = "GET_CLEAR_META"
    GET_CLEAR_META_RESPONSE = "GET_CLEAR_META_RESPONSE"
    CLEAR_META = "CLEAR_META"
    CLEAR_META_RESPONSE = "CLEAR_META_RESPONSE"

    # CHECK_CONSUMPTION
    CHECK_CONSUMPTION = "CHECK_CONSUMPTION"
    CONSUMPTION_RESPONSE = "CONSUMPTION_RESPONSE"

    # NOTIFY_DATA_UPDATE
    NOTIFY_DATA_UPDATE = "NOTIFY_DATA_UPDATE"
    NOTIFY_DATA_UPDATE_ACK = "NOTIFY_DATA_UPDATE_ACK"
    NOTIFY_DATA_UPDATE_ERROR = "NOTIFY_DATA_UPDATE_ERROR"


@dataclass
class ZMQServerInfo:
    role: TransferQueueRole
    id: str
    ip: str
    ports: dict[str, str]

    @classmethod
    def create(cls, role: TransferQueueRole, id: str, ip: str, ports: dict[str, str]) -> Self:
        return cls(role=role, id=id, ip=ip, ports=ports)

    def to_addr(self, port_name: str) -> str:
        return f"tcp://{self.ip}:{self.ports[port_name]}"

    def to_dict(self):
        return {
            "role": self.role,
            "id": self.id,
            "ip": self.ip,
            "ports": self.ports,
        }

    def __str__(self) -> str:
        return f"ZMQSocketInfo(role={self.role}, id={self.id}, ip={self.ip}, ports={self.ports})"


@dataclass
class ZMQMessage:
    request_type: ZMQRequestType
    sender_id: str
    receiver_id: str | None
    body: dict[str, Any]
    request_id: str
    timestamp: float

    @classmethod
    def create(
        cls,
        request_type: ZMQRequestType,
        sender_id: str,
        body: dict[str, Any],
        receiver_id: Optional[str] = None,
    ) -> "ZMQMessage":
        return cls(
            request_type=request_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            body=body,
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
        )

    def serialize(self) -> bytes:
        """Using pickle to serialize ZMQMessage objects"""
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes | list[bytes]):
        """Using pickle to deserialize ZMQMessage objects"""
        if isinstance(data, list):
            # Process multiple byte streams by deserializing each in sequence
            result = []
            for d in data:
                result.append(pickle.loads(d))
            return result
        else:
            # Single byte stream case
            return pickle.loads(data)


def get_free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def create_zmq_socket(
    ctx: zmq.Context,
    socket_type: Any,
    identity: Optional[bytes] = None,
) -> zmq.Socket:
    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)  # 0.5GB in bytes
    else:
        buf_size = -1  # Use system default buffer size

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)
    return socket
