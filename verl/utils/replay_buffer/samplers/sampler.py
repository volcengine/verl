from typing import Any, Callable, Iterator

"""
plugin based sampler design
"""


class Sampler:
    # TODO: have sample_keys as arguments
    def __init__(self, sample_keys={}) -> None:
        # each sample key has its own index

        # sample_keys is a set of attributes where sampler might slice and dice on
        # if we sample on a non specified sampling attribute, sampler will become extremely slow.
        # TODO: dedup the index if multiple samplers share the same attributes to be indexed
        self._index = {attribute: {} for attribute in sample_keys}

    def build_index(self, key, value):
        # build an index table which maps:
        # indexed key to key.
        pass

    def bind_task_processor(self, task_processor):
        # To merge with pending tasks in p0 queue during sampling
        self._task_processor = task_processor

    def get_index(self, index_name):
        return self._index[index_name]

    def remove_index(self, key):
        for val in self._index.values():
            val.pop(key, None)

    def sample(self) -> Iterator:
        raise NotImplementedError
