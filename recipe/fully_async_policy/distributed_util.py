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

import socket
from datetime import timedelta

from torch.distributed import TCPStore

from verl.utils.device import get_nccl_backend, is_npu_available
from verl.utils.net_utils import is_ipv6


class StatelessProcessGroupWrapper:
    """
    A wrapper class for distributed process groups that provides broadcast and allreduce methods.
    Supports both StatelessProcessGroup (from vLLM) and Ray Collective Group.
    """

    def __init__(self, process_group=None, communicator=None, is_ray=False):
        """
        Initialize the wrapper with a process group and optional communicator.

        Args:
            process_group: The underlying process group (either StatelessProcessGroup or Ray Collective Group)
            communicator: The communicator for StatelessProcessGroup (PyNcclCommunicator)
            is_ray: Whether this is a Ray Collective Group
        """
        self.process_group = process_group
        self.communicator = communicator
        self.is_ray = is_ray

    def broadcast(self, tensor, *args, src_rank, **kwargs):
        """
        Broadcast a tensor from the source rank to all other ranks.
        """
        if self.is_ray:
            import ray

            return ray.util.collective.broadcast(
                tensor, *args, src_rank=src_rank, group_name=self.process_group, **kwargs
            )
        else:
            return self.communicator.broadcast(tensor, *args, src=src_rank, **kwargs)

    def all_reduce(self, tensor, *args, **kwargs):
        """
        Allreduce a tensor across all ranks.
        """
        if self.is_ray:
            import ray

            return ray.util.collective.all_reduce(tensor, *args, group_name=self.process_group, **kwargs)
        else:
            return self.communicator.all_reduce(tensor, *args, **kwargs)

    @staticmethod
    def from_stateless_process_group(master_address, master_port, rank, world_size, device):
        """
        Initialize from a stateless process group (vLLM).

        vLLM provides `StatelessProcessGroup` to create a process group
        without considering the global process group in torch.distributed.
        It is recommended to create `StatelessProcessGroup`, and then initialize
        the data-plane communication (NCCL) between external (train processes)
        and vLLM workers.
        """
        # NOTE: If it is necessary to support weight synchronization with the sglang backend in the future,
        # the following can be used:
        # from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
        # from sglang.srt.distributed.utils import statelessprocessgroup
        if is_npu_available:
            from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator
        else:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        def create_process_group(
            host: str,
            port: int,
            rank: int,
            world_size: int,
            data_expiration_seconds: int = 3600,
            store_timeout: int = 300,
        ) -> "StatelessProcessGroup":
            """
            This is copied from vllm/distributed/utils.py:StatelessProcessGroup.create
            Modified to support ipv6 stateless communication groups."""
            launch_server = rank == 0
            if launch_server:
                # listen on the specified interface (instead of 0.0.0.0)
                if is_ipv6(master_address):
                    listen_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
                else:
                    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listen_socket.bind((host, port))
                listen_socket.listen()
                listen_fd = listen_socket.fileno()
            else:
                listen_socket = None
                listen_fd = None

            store = TCPStore(
                host_name=host,
                port=port,
                world_size=world_size,
                is_master=launch_server,
                timeout=timedelta(seconds=store_timeout),
                use_libuv=False,  # for now: github.com/pytorch/pytorch/pull/150215
                master_listen_fd=listen_fd,
            )

            return StatelessProcessGroup(
                rank=rank,
                world_size=world_size,
                store=store,
                socket=listen_socket,
                data_expiration_seconds=data_expiration_seconds,
            )

        pg = create_process_group(host=master_address, port=master_port, rank=rank, world_size=world_size)
        pynccl = PyNcclCommunicator(pg, device=device)

        return StatelessProcessGroupWrapper(communicator=pynccl, is_ray=False)

    @staticmethod
    def from_ray_collective_group(actor_rollout_workers, group_name):
        """
        Initialize a Ray Collective Group and return a wrapper instance.

        Args:
            actor_rollout_workers: The wokers to create the collective group
            group_name: The name of the Ray Collective Group
        """
        import ray

        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before creating a Ray Collective Group wrapper")

        # Initialize the Ray Collective Group if it doesn't exist
        from ray.util import collective

        collective.create_collective_group(
            actor_rollout_workers,
            len(actor_rollout_workers),
            list(range(0, len(actor_rollout_workers))),
            backend=get_nccl_backend(),
            group_name=group_name,
        )

        return StatelessProcessGroupWrapper(process_group=group_name, is_ray=True)
