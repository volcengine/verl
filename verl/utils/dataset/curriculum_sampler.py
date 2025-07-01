from collections import defaultdict
from typing import Any, Callable, Dict, Iterator
from collections.abc import Iterable, Iterator, Sequence, Sized

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, RandomSampler
from abc import ABC, abstractmethod


class AbstractCurriculumSampler(Sampler[int]):
    @abstractmethod
    def __init__(self, config):
        pass


class RandomCurriculumSampler(AbstractCurriculumSampler):
    def __init__(
        self,
        config: int,
    ):
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(1)
        sampler = RandomSampler(generator=train_dataloader_generator)
        self.sampler = sampler

    def __iter__(self):
        return self.sampler.__iter__()

    def __len__(self) -> int:
        return len(self.sampler)
