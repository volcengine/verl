import os
import torch.distributed.rpc as rpc
import multiprocessing as mp
import threading
import dill

# TODO: 重构LocalActor，使其更清真。

def _local_actor_runner(pickled_cls, args, kwargs, env_vars, input_queue, output_queue):
    if env_vars is not None:
        for k, v in env_vars.items():
            os.environ[k] = v
    cls = dill.loads(pickled_cls)
    inst = cls(*args, **kwargs)
    while True:
        cmd = input_queue.get()
        method_name, args, kwargs = cmd
        method = getattr(inst, method_name)
        result = method(*args, **kwargs)
        output_queue.put(result)

class LocalActor:
    def __init__(self, cls, args, kwargs, env_vars):
        ctx = mp.get_context('spawn')
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        pickled_cls = dill.dumps(cls)
        self.process = ctx.Process(target=_local_actor_runner, args=(pickled_cls, args, kwargs, env_vars, self.input_queue, self.output_queue), daemon=True)
        self.process.start()

    def run(self, method_name, args, kwargs):
        self.input_queue.put((method_name, args, kwargs))
        return self.output_queue.get()

class LocalActorManager:
    def __init__(self):
        self.lock = threading.Lock()

    def create_local_actor(self, cls, args, kwargs, env_vars, gpus):
        with self.lock:
            original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpus])
            actor = LocalActor(cls, args, kwargs, env_vars)
            if original_cuda_visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
        return actor

    def run(self, method_name, args, kwargs):
        method = getattr(self, method_name)
        return method(*args, **kwargs)

def _call_local_actor(actor_rref: rpc.RRef, method_name, args, kwargs):
    actor = actor_rref.local_value()
    return actor.run(method_name, args, kwargs)

def call_remote_actor(actor_rref: rpc.RRef, method_name, args, kwargs):
    return rpc.remote(actor_rref.owner(), _call_local_actor, args=(actor_rref, method_name, args, kwargs))

def rref_to_here(x):
    if isinstance(x, list):
        return [i.to_here() for i in x]
    else:
        return x.to_here()
