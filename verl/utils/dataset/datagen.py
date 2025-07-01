from abc import ABC, abstractmethod

from omegaconf import DictConfig
from torch.utils.data import Dataset


class AbstractDataGen(ABC):
    def __init__(self, config: DictConfig ):
        self.config = config

    @abstractmethod
    def generate(self, dataset: Dataset) -> None:
        """
        Generate method must be implemented by subclasses.
        Args:
            dataset: The dataset to generate from.
        Returns:
            Processed data or result as implemented by the subclass.
        """
        pass

class NoOpDataGen(AbstractDataGen):
    def __init__(self, config: DictConfig = None):
        super().__init__(config)

    def generate(self, dataset: Dataset) -> None:
        print ("NoOpDataGen: No operation performed on the dataset.")
        d = dataset.dataframe.select([0]) 
        # import ipdb; ipdb.set_trace()
        dataset.append_dataframe(d)  # No operation, just re-append the same data
        pass