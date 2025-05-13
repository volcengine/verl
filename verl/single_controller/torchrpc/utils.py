import os
import torch.distributed.rpc as rpc
import multiprocessing as mp
import threading
import cloudpickle

# TODO: 重构LocalActor，使其更清真。

def _local_actor_runner(pickled_args, input_queue, output_queue):
    cls, args, kwargs, env_vars = cloudpickle.loads(pickled_args)
    if env_vars is not None:
        for k, v in env_vars.items():
            os.environ[k] = v
    inst = cls(*args, **kwargs)
    while True:
        pickled_cmd = input_queue.get()
        method_name, args, kwargs = cloudpickle.loads(pickled_cmd)
        method = getattr(inst, method_name)
        result = method(*args, **kwargs)
        output_queue.put(result)

class LocalActor:
    def __init__(self, cls, args, kwargs, env_vars):
        ctx = mp.get_context('spawn')
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.process = ctx.Process(target=_local_actor_runner, args=(cloudpickle.dumps((cls, args, kwargs, env_vars)), self.input_queue, self.output_queue), daemon=True)
        self.process.start()

    def run(self, method_name, args, kwargs):
        self.input_queue.put(cloudpickle.dumps((method_name, args, kwargs)))
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
    
    def visible_gpus(self):
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is None:
            import torch
            return list(range(torch.cuda.device_count()))
        else:
            return [int(i) for i in cuda_visible_devices.split(',')]


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
