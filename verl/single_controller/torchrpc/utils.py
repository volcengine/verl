import os
from multiprocessing import Process, Queue
import torch.distributed.rpc as rpc
import multiprocessing as mp

def _local_actor_runner(cls, args, kwargs, input_queue, output_queue):
    inst = cls(*args, **kwargs)
    while True:
        cmd = input_queue.get()
        if cmd is None:
            break
        method_name, args, kwargs = cmd
        method = getattr(inst, method_name)
        result = method(*args, **kwargs)
        output_queue.put(result)

class RemoteActor:
    def __init__(self, cls, args, kwargs):
        ctx = mp.get_context('spawn')
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.process = ctx.Process(target=_local_actor_runner, args=(cls, args, kwargs, self.input_queue, self.output_queue))
        self.process.start()

    def run_method(self, method_name, args, kwargs):
        self.input_queue.put((method_name, args, kwargs))
        return self.output_queue.get()
    
    def __del__(self):
        self.input_queue.put(None)
        self.process.join()

remote_actors = None

def create_remote_actor(cls, args, kwargs, env_vars=None):
    global remote_actors
    if not remote_actors:
        remote_actors = []
    if env_vars is not None:
        for k, v in env_vars.items():
            os.environ[k] = v
    actor = RemoteActor(cls, args, kwargs)
    remote_actors.append(actor)
    return actor

def stop_remote_actors():
    global remote_actors
    if remote_actors:
        for actor in remote_actors:
            del actor
        remote_actors = None

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
