from itertools import cycle

import ray


@ray.remote
class ExecutorWorker:
    def execute(self, fn, /, *args, **kwargs):
        return fn(*args, **kwargs)


class RayActorExecutor:
    def __init__(self, max_workers=1):
        self.workers = [ExecutorWorker.options(scheduling_strategy="SPREAD").remote() for _ in range(max_workers)]
        self.worker_pointer = cycle(range(max_workers))

    def submit(self, fn, /, *args, **kwargs):
        worker_idx = next(self.worker_pointer)
        worker = self.workers[worker_idx]
        obj_ref = worker.execute.remote(fn, *args, **kwargs)
        return obj_ref.future()
