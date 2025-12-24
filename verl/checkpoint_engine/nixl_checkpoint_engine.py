# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import pickle
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import AsyncGenerator, Generator

import nixl._api as nixl_api
import nixl._bindings as nixl_bindings
import ray
import torch

from verl.checkpoint_engine.base import CheckpointEngine, TensorMeta
from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NixlAgent:
    def __init__(self, agent: nixl_api.nixl_agent):
        self.agent = agent
        self.notifications: dict[str, deque[bytes]] = defaultdict(deque)

    def __getattr__(self, name):
        attr = getattr(self.agent, name)

        if callable(attr):

            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs)

            return wrapper
        else:
            return attr

    async def get_notification(self, remote_name: str) -> bytes:
        while len(self.notifications[remote_name]) == 0:
            notifs = self.agent.get_new_notifs()
            for remote_name, notif in notifs.items():
                self.notifications[remote_name].extend(notif)
            await asyncio.sleep(0.1)
        return self.notifications[remote_name].popleft()


@dataclass
class NixlAgentAddress:
    """Address for a Nixl agent."""

    name: str
    ip: str
    port: int


class ReadableOperation:
    """Encapsulates a readable operation to remote agent.
       1. send metadata to remote agent
       2. wait until remote agent read complete.

    Args:
        agent (NixlAgent): The Nixl agent.
        remote_agent (str): The name of the remote agent.
        local_descs (nixl_bindings.nixlXferDList): The local transfer descriptors.
        metadata (dict): Metadata for the read operation.
    """

    def __init__(self, agent: NixlAgent, remote_agent: str, local_descs: nixl_bindings.nixlXferDList, metadata: dict):
        self.agent = agent
        self.remote_agent = remote_agent
        self.local_descs = local_descs
        self.notify_key = uuid.uuid4().bytes
        msg = pickle.dumps({"notify_key": self.notify_key, "remote_descs": self.local_descs, **metadata})
        self.agent.send_notif(self.remote_agent, msg)

    async def wait_for_complete(self):
        """Block until remote agent read complete."""
        notification = await self.agent.get_notification(self.remote_agent)
        assert self.notify_key == notification, f"Notify key {self.notify_key} not equal to {notification}"
        logger.debug(f"ReadableOperation {self.notify_key} complete")


class ReadOperation:
    """Encapsulates a read operation from remote agent.
    1. read medata from remote agent
    2. start read transfer operation
    3. wait until read complete

    Args:
        agent (NixlAgent): The Nixl agent.
        remote_agent (str): The name of the remote agent.
        local_descs (nixl_bindings.nixlXferDList): The local transfer descriptors.
    """

    def __init__(self, agent: NixlAgent, remote_agent: str, local_descs: nixl_bindings.nixlXferDList):
        self.agent = agent
        self.remote_agent = remote_agent
        self.local_descs = local_descs
        self.remote_descs = None
        self.xfer_handle = None
        self.notify_key = None

    async def read_metadata(self) -> dict:
        """Block until the remote agent sends the metadata.

        Returns:
            dict: Metadata from the remote agent.
        """
        notification = await self.agent.get_notification(self.remote_agent)
        metadata = pickle.loads(notification)
        self.remote_descs = metadata.pop("remote_descs")
        self.notify_key = metadata.pop("notify_key")
        return metadata

    def begin_read(self):
        """Start the read operation."""
        assert self.remote_descs is not None and self.notify_key is not None
        self.xfer_handle = self.agent.initialize_xfer(
            "READ", self.local_descs, self.remote_descs, self.remote_agent, self.notify_key
        )
        state = self.agent.transfer(self.xfer_handle)
        assert state != "ERR", f"Read from {self.remote_agent} got to {state} state."

    async def wait_for_complete(self):
        """Block until the read operation complete."""
        while True:
            state = self.agent.check_xfer_state(self.xfer_handle)
            if state == "ERR":
                logger.error(f"Read from {self.remote_agent} got to {state} state.")
                exit(-1)
            elif state == "DONE":
                break
            else:
                await asyncio.sleep(0.1)
        self.agent.release_xfer_handle(self.xfer_handle)
        logger.debug(f"ReadOperation read data {self.notify_key} from {self.remote_agent} complete")


class NIXLCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size: int, device: str):
        self.bucket_size = bucket_size
        self.device = device
        self.agent_name = str(uuid.uuid4())
        self.ip = ray.util.get_node_ip_address().strip("[]")

        self.listen_port, self.listen_sock = get_free_port(self.ip)
        self.agent = NixlAgent(
            nixl_api.nixl_agent(self.agent_name, nixl_api.nixl_agent_config(True, True, self.listen_port))
        )

    def get_metadata(self) -> NixlAgentAddress:
        return NixlAgentAddress(self.agent_name, self.ip, self.listen_port)

    def setup(self, rank: int, world_size: int, prev_address: NixlAgentAddress, next_address: NixlAgentAddress):
        """Get the full metadata of the send and recv agents.

        Returns:
            A tuple of the send and recv agent metadata.
        """
        self.rank = rank
        self.world_size = world_size
        self.prev_agent = None
        self.next_agent = None

        # create and register memory for send and recv bucket.
        self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device)
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device)
        self.send_reg_descs = self.agent.register_memory(self.send_buf)
        self.recv_reg_descs = self.agent.register_memory(self.recv_buf)
        self.send_descs = self.agent.get_xfer_descs(self.send_buf)
        self.recv_descs = self.agent.get_xfer_descs(self.recv_buf)

        # check prev agent metadata ready
        if prev_address is not None:
            self.prev_agent = prev_address.name
            self.agent.fetch_remote_metadata(self.prev_agent, prev_address.ip, prev_address.port)
            self.agent.send_local_metadata(prev_address.ip, prev_address.port)
            while True:
                if self.agent.check_remote_metadata(self.prev_agent):
                    break
                time.sleep(1)

        # check next agent metadata ready
        if next_address is not None:
            self.next_agent = next_address.name
            while True:
                if self.agent.check_remote_metadata(self.next_agent):
                    break
                time.sleep(1)

        logger.info(
            f"Setup rank: {self.rank}, world_size: {self.world_size}, "
            f"prev_agent: {self.prev_agent}, next_agent: {self.next_agent}"
        )

    def tear_down(self):
        """Tear down the communication with the previous and next agent."""
        if self.prev_agent:
            self.agent.remove_remote_agent(self.prev_agent)
        if self.next_agent:
            self.agent.remove_remote_agent(self.next_agent)

        self.agent.deregister_memory(self.send_reg_descs)
        self.agent.deregister_memory(self.recv_reg_descs)
        self.send_buf = None
        self.recv_buf = None
        self.send_reg_descs = None
        self.recv_reg_descs = None
        self.send_descs = None
        self.recv_descs = None

        self.rank = None
        self.world_size = None
        self.prev_agent = None
        self.next_agent = None

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        assert self.next_agent is not None, "Next agent is not set."
        send_buf = self.send_buf
        send_descs = self.send_descs

        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        for name, weight in weights:
            # fill the tensor bucket
            if offset + weight.nbytes > self.bucket_size:
                torch.cuda.synchronize()

                # send bucket meta to next agent
                readable_op = ReadableOperation(
                    self.agent, self.next_agent, send_descs, {"bucket_meta": bucket_meta, "is_last": False}
                )
                await readable_op.wait_for_complete()

                # reset bucket meta and offset
                bucket_meta = {}
                offset = 0

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
            )

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            send_buf[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
            offset += weight.nbytes

        # send last bucket meta to next agent
        torch.cuda.synchronize()
        readable_op = ReadableOperation(
            self.agent, self.next_agent, send_descs, {"bucket_meta": bucket_meta, "is_last": True}
        )
        await readable_op.wait_for_complete()
        logger.info(f"Rank {self.rank} send weights done!")

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        assert self.prev_agent is not None, "Previous agent is not set."
        send_buf, recv_buf = self.send_buf, self.recv_buf
        send_descs, recv_descs = self.send_descs, self.recv_descs

        # receive first bucket from previous agent
        read_op = ReadOperation(self.agent, self.prev_agent, recv_descs)
        metadata = await read_op.read_metadata()
        read_op.begin_read()
        await read_op.wait_for_complete()

        # swap send and recv buf
        send_buf, recv_buf = recv_buf, send_buf
        send_descs, recv_descs = recv_descs, send_descs
        while not metadata["is_last"]:
            # 1. send bucket to next agent
            readable_op = None
            if self.next_agent is not None:
                readable_op = ReadableOperation(
                    self.agent,
                    self.next_agent,
                    send_descs,
                    metadata,
                )

            # 2. receive bucket from previous agent
            read_op = ReadOperation(self.agent, self.prev_agent, recv_descs)
            next_metadata = await read_op.read_metadata()
            read_op.begin_read()

            # 3. yield tensor from send_buf
            for name, meta in metadata["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = send_buf[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            # 4. wait for next agent read complete and read from previous agent complete
            if readable_op is not None:
                await readable_op.wait_for_complete()
            await read_op.wait_for_complete()

            # 5. swap send and recv buf
            metadata = next_metadata
            send_buf, recv_buf = recv_buf, send_buf
            send_descs, recv_descs = recv_descs, send_descs

        # send last bucket to next agent
        readable_op = None
        if self.next_agent is not None:
            readable_op = ReadableOperation(
                self.agent,
                self.next_agent,
                send_descs,
                metadata,
            )

        # yield tensor from send_buf
        for name, meta in metadata["bucket_meta"].items():
            dtype, shape = meta["dtype"], meta["shape"]
            size = dtype.itemsize * shape.numel()
            tensor = send_buf[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
            yield name, tensor

        # wait for next agent read complete
        if readable_op is not None:
            await readable_op.wait_for_complete()
        logger.info(f"Rank {self.rank} receive weights done!")
