from typing import List
import os
import torch.distributed.rpc as rpc
from verl.single_controller.torchrpc.utils import LocalActorManager, RemoteActor, call_remote_actor

class Node:
    def __init__(self, name):
        self.name = name
        self.local_actor_manager_rref = rpc.remote(self.name, LocalActorManager)
        self.dispatched_resources = []
        # TODO: 初始化获取可用资源
        self.gpus = [0, 1, 2, 3, 4, 5, 6, 7]
        self.actor_gpu_usage = {} # {RemoteActor: gpus}
    
    def create_actor(self, cls, args, kwargs, env_vars, gpus) -> RemoteActor:
        rref = call_remote_actor(self.local_actor_manager_rref, 'create_local_actor', (cls, args, kwargs, env_vars, gpus))
        return RemoteActor(self, rref)
    
    def dispatch(self, gpu) -> 'NodeResource':
        assert len(self.gpus) >= gpu
        self.gpus.sort()
        gpus = self.gpus[:gpu]
        self.gpus = self.gpus[gpu:]
        ret = NodeResource(self, gpus)
        self.dispatched_resources.append(ret)
        return ret
    
    def recycle(self, resource: 'NodeResource'):
        self.gpus.extend(resource.gpus)
        self.dispatched_resources.remove(resource)


class NodeResource:
    def __init__(self, node: Node, gpus: List[int]):
        self.node = node
        self.gpus = gpus
        self.actors = []
    
    def create_actor(self, cls, args, kwargs, env_vars, num_gpus) -> RemoteActor:
        gpus = self.gpus[:num_gpus]
        self.gpus = self.gpus[num_gpus:]
        actor = self.node.create_actor(cls, args, kwargs, env_vars, gpus)
        self.actors.append(actor)
        return actor
    
    def recycle(self, actor: RemoteActor):
        self.actors.remove(actor)
        self.gpus.extend(actor.gpus)
    
    def __del__(self):
        self.node.recycle(self)


class NodeManager:
    def __init__(self):
        world_size = int(os.environ['TORCHRPC_WORLD_SIZE'])
        self.nodes = [Node(f'torchrpc_worker{i}') for i in range(world_size)]

    def dispatch(self, gpu) -> NodeResource:
        ret = None
        for node in self.nodes:
            if len(node.gpus) > gpu:
                if ret is None or len(node.gpus) < len(ret.gpus):
                    ret = node
        assert ret is not None
        return ret.dispatch(gpu)
