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
"""
the class for Worker
"""
import os
import socket
from dataclasses import dataclass
from .decorator import register, Dispatch, Execute


@dataclass
class DistRankInfo:
    tp_rank: int
    dp_rank: int
    pp_rank: int


@dataclass
class DistGlobalInfo:
    tp_size: int
    dp_size: int
    pp_size: int


class WorkerHelper:

    def _get_node_ip(self):

        def get_node_ip_by_sdk():
            if os.getenv("WG_BACKEND", None) == "ray":
                import ray
                return ray._private.services.get_node_ip_address()
            else:
                raise NotImplementedError("WG_BACKEND now just support ray mode.")

        host_ipv4 = os.getenv("MY_HOST_IP", None)
        host_ipv6 = os.getenv("MY_HOST_IPV6", None)
        host_ip_by_env = host_ipv4 or host_ipv6
        host_ip_by_sdk = get_node_ip_by_sdk()

        host_ip = host_ip_by_env or host_ip_by_sdk
        return host_ip

    def _get_free_port(self):
        with socket.socket() as sock:
            sock.bind(('', 0))
            return sock.getsockname()[1]

    def get_availale_master_addr_port(self):
        return self._get_node_ip(), str(self._get_free_port())

    def _get_pid(self):
        return


class WorkerMeta:
    keys = [
        "WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES"
    ]

    def __init__(self, store) -> None:
        self._store = store

    def to_dict(self):
        return {f"_{key.lower()}": self._store.get(f"_{key.lower()}", None) for key in WorkerMeta.keys}


# we assume that in each WorkerGroup, there is a Master Worker
class Worker(WorkerHelper):
    """A (distributed) worker."""

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)

        # note that here we use int to distinguish
        disable_worker_init = int(os.environ.get('DISABLE_WORKER_INIT', 0))
        if disable_worker_init:
            return instance

        rank = os.environ.get("RANK", None)
        worker_group_prefix = os.environ.get("WG_PREFIX", None)

        # when decorator @ray.remote applies, __new__ will be called while we don't want to apply _configure_before_init
        if None not in [rank, worker_group_prefix] and 'ActorClass(' not in cls.__name__:
            instance._configure_before_init(f"{worker_group_prefix}_register_center", int(rank))

        return instance

    def _configure_before_init(self, register_center_name: str, rank: int):
        assert isinstance(rank, int), f"rank must be int, instead of {type(rank)}"

        if rank == 0:
            master_addr, master_port = self.get_availale_master_addr_port()
            rank_zero_info = {
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            }

            if os.getenv("WG_BACKEND", None) == "ray":
                from verl.single_controller.base.register_center.ray import create_worker_group_register_center
                self.register_center = create_worker_group_register_center(name=register_center_name,
                                                                           info=rank_zero_info)

            os.environ.update(rank_zero_info)

    def __init__(self, cuda_visible_devices=None) -> None:
        # construct a meta from envrionment variable. Note that the import must be inside the class because it is executed remotely
        import os
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        self._rank = rank
        self._world_size = world_size

        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]

        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        store = {
            '_world_size': world_size,
            '_rank': rank,
            '_local_world_size': local_world_size,
            '_local_rank': local_rank,
            '_master_addr': master_addr,
            '_master_port': master_port
        }
        if cuda_visible_devices is not None:
            store['_cuda_visible_devices'] = cuda_visible_devices

        meta = WorkerMeta(store=store)
        self._configure_with_meta(meta=meta)

    def _configure_with_meta(self, meta: WorkerMeta):
        """
        This function should only be called inside by WorkerGroup
        """
        assert isinstance(meta, WorkerMeta)
        self.__dict__.update(meta.to_dict())  # this is hacky
        # print(f"__dict__: {self.__dict__}")
        for key in WorkerMeta.keys:
            val = self.__dict__.get(f"_{key.lower()}", None)
            if val is not None:
                # print(f"set {key} to {val}")
                os.environ[key] = str(val)
        os.environ["REDIS_STORE_SERVER_HOST"] = str(self._master_addr).replace("[", "").replace(
            "]", "") if self._master_addr else ""

    def get_master_addr_port(self):
        return self._master_addr, self._master_port

    def get_cuda_visible_devices(self):
        import os
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        return cuda_visible_devices

    @property
    def world_size(self):
        return self._world_size

    @property
    def rank(self):
        return self._rank

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    def execute_with_func_generator(self, func, *args, **kwargs):
        ret_proto = func(self, *args, **kwargs)
        return ret_proto

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def execute_func_rank_zero(self, func, *args, **kwargs):
        result = func(*args, **kwargs)
        return result
