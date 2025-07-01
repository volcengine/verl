from abc import abstractmethod
from collections.abc import Sized

import torch
from omegaconf import DictConfig
from torch.utils.data import RandomSampler, Sampler


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
