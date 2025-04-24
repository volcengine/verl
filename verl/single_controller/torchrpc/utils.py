import os
from multiprocessing import Process, Queue
import torch.distributed.rpc as rpc
import multiprocessing as mp

def _local_actor_runner(cls, args, kwargs, queue):
    inst = cls(*args, **kwargs)
    while True:
        cmd = queue.get()
        if cmd is None:
            break
        out_queue, method_name, args, kwargs = cmd
        method = getattr(inst, method_name)
        result = method(*args, **kwargs)
        out_queue.put(result)

class RemoteActor:
    def __init__(self):
        pass

    def init(self, cls, args, kwargs):
        ctx = mp.get_context('spawn')
        self.queue = ctx.Queue()
        self.process = ctx.Process(target=_local_actor_runner, args=(cls, args, kwargs, self.queue))
        self.process.start()

    def run_method(self, method_name, *args, **kwargs):
        out_queue = Queue()
        self.queue.put((out_queue, method_name, args, kwargs))
        return out_queue.get()

    def terminate(self):
        self.queue.put(None)
        self.process.join()

def create_remote_actor(cls, args, kwargs, env_vars=None):
    if env_vars is not None:
        for k, v in env_vars.items():
            os.environ[k] = v
    actor = RemoteActor()
    actor.init(cls, args, kwargs)
    return actor

def _remote_actor_call_local(actor_rref, method_name, args, kwargs):
    actor = actor_rref.local_value()
    return actor.run_method(method_name, args, kwargs)

def remote_actor_call(actor_rref, method_name, args, kwargs):
    return rpc.remote(actor_rref.owner(), _remote_actor_call_local, args=(actor_rref, method_name, args, kwargs))

def rref_to_here(x):
    if isinstance(x, list):
        return [i.to_here() for i in x]
    else:
        return x.to_here()