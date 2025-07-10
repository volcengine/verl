import random

from verl.utils.replay_buffer.replay_buffer_client import ReplayBufferClient

"""
The replay buffer backed by an in-mem dict.
It's NOT thread safe.
"""


class VanillaReplayBufferClient(ReplayBufferClient):
    def __init__(self):
        self.__pool = dict()

    def push(self, key, batches):
        if not isinstance(batches, list):
            batches = [batches]
        if key not in self.__pool:
            self.__pool[key] = []
        for batch in batches:
            self.__pool[key].append(batch)

    def get(self, key: str):
        return self.__pool.get(key, None)

    def sample(self):
        keys = list(self.__pool.keys())
        while keys:  # raises StopIteration on every next() if keys is empty
            random_key = random.choice(keys)
            yield random_key

    def delete(self, key: str):
        self.__pool.pop(key)
