import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Callable

import ray
from torch import TypeVar

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_TOOLS_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        # this only used for observalability
        self.current_count = 0
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count


class ExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10, rate_limiter_name="rate-limiter"):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        self.rate_limiter_name = rate_limiter_name

    def _init_rate_limit(self, rate_limit):
        # TODO validation for rate_limit
        # A Singleton Rate Limitor
        return TokenBucketWorker.options(name=self.rate_limiter_name, get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        with ExitStack() as stack:
            stack.callback(self.rate_limit_worker.release.remote)
            ray.get(self.rate_limit_worker.acquire.remote())
            try:
                return fn(*fn_args, **fn_kwargs)
            except Exception as e:
                # TODO we should make this available to the tool caller
                logger.warning(f"Error when executing code: {e}")


def init_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode, rate_limiter_name="rate-limiter"):
    if mode == PoolMode.ThreadMode:
        return ray.remote(ExecutionWorker).options(max_concurrency=num_workers).remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit, rate_limiter_name=rate_limiter_name)
    else:
        raise NotImplementedError("Process mode is not implemented yet")
        # return ray.util.multiprocessing.Pool(processes=num_workers)
