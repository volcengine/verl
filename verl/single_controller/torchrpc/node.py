from typing import List
import os
import torch.distributed.rpc as rpc
from verl.single_controller.torchrpc.utils import LocalActorManager, call_remote_actor

class Node:
    def __init__(self, name):
        self.name = name
        self.local_actor_manager_rref = rpc.remote(self.name, LocalActorManager)
        # self.dispatched_resources = []
        # TODO: 初始化获取可用资源
        self.gpus = [0, 1, 2, 3, 4, 5, 6, 7]

    def dispatch(self, gpu) -> 'NodeResource':
        assert len(self.gpus) >= gpu
        self.gpus.sort()
        gpus = self.gpus[:gpu]
        self.gpus = self.gpus[gpu:]
        ret = NodeResource(self, gpus)
        # self.dispatched_resources.append(ret)
        return ret

    def recycle(self, resource: 'NodeResource'):
        self.gpus.extend(resource.gpus)
        # self.dispatched_resources.remove(resource)


class NodeResource:
    def __init__(self, node: Node, gpus: List[int]):
        self.node = node
        self.gpus = gpus
        self.actors = []
    
    def create_actor(self, cls, args, kwargs, env_vars, gpus) -> rpc.RRef:
        assert all(i in self.gpus for i in gpus)
        rref = call_remote_actor(self.node.local_actor_manager_rref, 'create_local_actor', (cls, args, kwargs, env_vars, gpus), {})
        self.actors.append(rref)
        return rref

    def __del__(self):
        self.node.recycle(self)


class NodeManager:
    def __init__(self):
        self._initiated = False
    
    def init(self):
        if not self._initiated:
            self._initiated = True
            world_size = int(os.environ['TORCHRPC_WORLD_SIZE'])
            self.nodes = [Node(f'torchrpc_worker{i}') for i in range(world_size)]

    def dispatch(self, gpu) -> NodeResource:
        # TODO: CPU only下全部派给一个node
        ret = None
        for node in self.nodes:
            if len(node.gpus) >= gpu:
                if ret is None or len(node.gpus) < len(ret.gpus):
                    ret = node
        assert ret is not None
        return ret.dispatch(gpu)

node_manager = NodeManager()