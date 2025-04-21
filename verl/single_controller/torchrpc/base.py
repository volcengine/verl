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

import os
import torch
from typing import List, Dict, Any

import torch.distributed.rpc as rpc
from verl.single_controller.base import WorkerGroup, ResourcePool, ClassWithInitArgs, Worker

def get_random_string(length):
    import random
    import string
    letters_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_digits) for _ in range(length))

def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    def func(*args, **kwargs):
        args, kwargs = dispatch_fn(self, *args, **kwargs)
        output = execute_fn(method_name, *args, **kwargs)
        if blocking:
            if isinstance(output, List):
                new_output = []
                for o in output:
                    if isinstance(o, rpc.PyRRef):
                        o = o.to_here()
                    new_output.append(o)
                output = new_output
            elif isinstance(output, rpc.PyRRef):
                    output = output.to_here()
        output = collect_fn(self, output)
        return output

    return func

def _call_local_method(rref, method_name, args, kwargs):
    obj = rref.local_value()
    method = getattr(obj, method_name)
    return method(*args, **kwargs)

def _call_remote_method(rref, method_name, args, kwargs):
    return rpc.remote(rref.owner(), _call_local_method, args=(rref, method_name, args, kwargs))

def _create_local_instance(cls, args, kwargs, env_vars):
    if env_vars:
        for k, v in env_vars.items():
            os.environ[k] = v
    return cls(*args, **kwargs)

class Node:
    def __init__(self, name, total_cpu=-1, total_gpu=-1, used_cpu=0, used_gpu=0):
        self.name = name
        self.total_cpu = total_cpu if total_cpu != -1 else rpc.remote(name, os.cpu_count).to_here()
        self.total_gpu = total_gpu if total_gpu != -1 else rpc.remote(name, torch.cuda.device_count).to_here()
        self.used_cpu = used_cpu
        self.used_gpu = used_gpu

global_nodes = None

def get_global_nodes():
    global global_nodes
    if not global_nodes:
        global_nodes = [Node('header'), Node('worker1')]
    return global_nodes

def get_best_node(cpu, gpu):
    nodes = get_global_nodes()
    ret = None
    for node in nodes:
        if node.total_cpu - node.used_cpu >= cpu and node.total_gpu - node.used_gpu >= gpu:
            if ret is None or node.total_gpu - node.used_gpu < ret.total_gpu - ret.used_gpu:
                ret = node
    return ret

# name_prefix & detached 未实现
class TorchRPCResourcePool(ResourcePool):
    def __init__(self,
                 process_on_nodes: List[int] = None,
                 use_gpu: bool = True,
                 name_prefix: str = "",
                 max_colocate_count: int = 5,
                 detached=False) -> None:
        super().__init__(process_on_nodes, max_colocate_count)
        self.use_gpu = use_gpu
        self.name_prefix = name_prefix
        self.nodes = None
        self.detached = detached

    def get_nodes(self):
        if self.nodes is not None:
            return self.nodes
        
        self.nodes = []
        for process_count in self._store:
            cpu = self.max_collocate_count * process_count
            gpu = process_count if self.use_gpu else 0
            node = get_best_node(cpu, gpu)
            if node is None:
                raise Exception("No node found")
            node.used_cpu += cpu
            node.used_gpu += gpu
            self.nodes.append(node)
        return self.nodes

class TorchRPCClassWithInitArgs(ClassWithInitArgs):
    def __init__(self, cls, *args, **kwargs) -> None:
        super().__init__(cls, *args, **kwargs)
        self._options = {}
        self._additional_resource = {}
        self.env_vars = {}

    def set_additional_resource(self, additional_resource):
        self._additional_resource = additional_resource

    def update_options(self, options: Dict):
        self._options.update(options)

    def update_env_vars(self, env_vars: Dict):
        self.env_vars.update(env_vars)

    def __call__(self,
                 node: Node,
                 use_gpu: bool = True,
                 num_gpus=1,
                 sharing_with=None) -> Any:
        # options = self._options.copy()
        # if use_gpu:
        #     options["num_gpus"] = num_gpus

        # if len(self._additional_resource) >= 1:
        #     for k, v in self._additional_resource.items():
        #         options[k] = v

        return rpc.remote(node.name, _create_local_instance, args=(self.cls, self.args, self.kwargs, self.env_vars))

class TorchRPCWorkerGroup(WorkerGroup):

    def __init__(self,
                 resource_pool: TorchRPCResourcePool = None,
                 cls_with_init: TorchRPCClassWithInitArgs = None,
                 bin_pack: bool = True,
                 name_prefix: str = None,
                 detached=False,
                 worker_names=None,
                 **kwargs) -> None:
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.cls_with_init = cls_with_init
        # 名字怎么用？
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix

        if worker_names is not None:
            # 我们的worker是无名的
            # _is_init_with_detached_workers -> resource_pool is None
            assert self._is_init_with_detached_workers
            self._worker_names = worker_names

        if self._is_init_with_detached_workers:
            # 我们有 detached worker吗？
            self._init_with_detached_workers(worker_names=worker_names)
        else:
            self._init_with_resource_pool(resource_pool=resource_pool,
                                          cls_with_init=cls_with_init,
                                          bin_pack=bin_pack,
                                          detached=detached)

        if cls_with_init is not None:
            # 研究下如何绑定
            self._bind_worker_method(self.cls_with_init.cls, func_generator)

    def _is_worker_alive(self, worker):
        return True # TODO

    def _init_with_detached_workers(self, worker_names):
        raise NotImplementedError # TODO

    def _init_with_resource_pool(self, resource_pool, cls_with_init, bin_pack, detached):
        use_gpu = resource_pool.use_gpu

        nodes = resource_pool.get_nodes()
        self._world_size = resource_pool.world_size

        rank = -1
        for node_idx, node in enumerate(nodes):
            local_world_size = resource_pool.store[node_idx]
            for local_rank in range(local_world_size):
                rank += 1
                
                # we pass in environment variable at option so that Worker can use environment variable to set
                env_vars = {
                    'WORLD_SIZE': str(self._world_size),
                    'RANK': str(rank),
                    # TODO:前缀
                    # 'WG_PREFIX': self.name_prefix,
                    'WG_BACKEND': 'torchrpc',
                    'TORCHRPC_LOCAL_WORLD_SIZE': str(local_world_size),
                    'TORCHRPC_LOCAL_RANK': str(local_rank),
                }
                # if rank != 0:
                #     env_vars['MASTER_ADDR'] = self._master_addr
                #     env_vars['MASTER_PORT'] = self._master_port

                cls_with_init.update_env_vars(env_vars)
                worker = cls_with_init(node, use_gpu, )
                self._workers.append(worker)

                # if rank == 0:
                #     # TODO: GET ADDR
                #     self._master_addr = worker.get_master_addr()
                #     self._master_port = worker.get_master_port()

    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        return _call_remote_method(self._workers[0], method_name, args, kwargs)

    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        return self.execute_rank_zero_async(method_name, *args, **kwargs).to_here()

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all_async(self, method_name: str, *args, **kwargs):
        # Here, we assume that if all arguments in args and kwargs are lists, and their lengths match len(self._workers),
        # we'll distribute each element in these lists to the corresponding worker
        # print(f"execute_all_async: method {method_name}({args}, {kwargs})")
        length = len(self._workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                # print(f"splitting args and kwargs into {length} shards")
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    result.append(_call_remote_method(self._workers[i], method_name, sliced_args, sliced_kwargs))
                return result

        return [_call_remote_method(worker, method_name, args, kwargs) for worker in self._workers]

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        ret = self.execute_all_async(method_name, *args, **kwargs)
        return [r.to_here() for r in ret]

    def execute_all(self, method_name: str, *args, **kwargs):
        return self.execute_all_async(method_name, *args, **kwargs)

from verl.single_controller.base.decorator import MAGIC_ATTR
from verl.single_controller.ray.base import _bind_workers_method_to_parent

def create_colocated_worker_cls(class_dict: dict[str, TorchRPCClassWithInitArgs]):
    """
    This function should return a class instance that delegates the calls to every 
    cls in cls_dict
    """
    cls_dict = {}
    init_args_dict = {}
    worker_cls = None
    for key, cls in class_dict.items():
        if worker_cls == None:
            worker_cls = cls.cls.__base__
        else:
            assert worker_cls == cls.cls.__base__, \
                'the worker class should be the same when share the same process'
        cls_dict[key] = cls.cls
        init_args_dict[key] = {'args': cls.args, 'kwargs': cls.kwargs}

    assert cls_dict.keys() == init_args_dict.keys()

    # TODO: create a class with customizable name
    class WorkerDict(worker_cls):

        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                self.worker_dict[key] = user_defined_cls(*init_args_dict[key].get('args', ()),
                                                         **init_args_dict[key].get('kwargs', {}))

    # now monkey-patch the methods from inner class to WorkerDict
    for key, user_defined_cls in cls_dict.items():
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    remote_cls = TorchRPCClassWithInitArgs(cls=WorkerDict)
    return remote_cls


def rref_to_here(x):
    if isinstance(x, list):
        return [i.to_here() for i in x]
    else:
        return x.to_here()