from omegaconf import DictConfig
from typing import Iterator
from collections.abc import Sized


import torch
from torch.utils.data import Sampler, RandomSampler
from abc import abstractmethod


class AbstractCurriculumSampler(Sampler[int]):
    @abstractmethod
    def __init__(
        self,
        data_source: Sized,
        config: DictConfig,
    ):
        pass


class RandomCurriculumSampler(AbstractCurriculumSampler):
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(1)
        sampler = RandomSampler(data_source=data_source)
        self.sampler = sampler

    def __iter__(self):
        return self.sampler.__iter__()

    def __len__(self) -> int:
        return len(self.sampler)
