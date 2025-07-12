import random
from typing import Iterator

from verl.utils.replay_buffer.persistable_replay_buffer_client import get_key_index_name
from verl.utils.replay_buffer.samplers.sampler import Sampler


class UniformKeySampler(Sampler):
    def __init__(self) -> None:
        super().__init__({get_key_index_name()})

    def build_index(self, key, value):
        # build an index table which maps:
        # indexed key to key.
        self._index[get_key_index_name()][key] = key

    def sample(self) -> Iterator:
        self._task_processor.acquire_q_lock()

        key_index = self._index[get_key_index_name()]
        keys = set(key_index.keys())
        # merge with pending tasks in p0 queue to get updates for all keys
        keys = list(self._task_processor.merge_pending_keys(keys))
        self._task_processor.release_q_lock()
        while keys:  # raises StopIteration on every next() if keys is empty
            random_key = random.choice(keys)
            yield random_key
