# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import datasets
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset import RLHFDataset
from verl.utils.import_utils import load_extern_type


class AbstractDataGen(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def generate(self, dataset: Dataset) -> datasets.Dataset:
        """
        Generate method must be implemented by subclasses.
        Args:
            dataset: The dataset to generate from.
        Returns:
            Processed data or result as implemented by the subclass.
        """
        pass


class NoOpDataGen(AbstractDataGen):
    """
    A noop data gen class that only reappends the first datapoint.
    This class is useful as a placeholder and testing.
    """

    def __init__(self, config: DictConfig = None):
        super().__init__(config)

    def generate(self, dataset: Dataset) -> datasets.Dataset:
        print("NoOpDataGen: No operation performed on the dataset.")
        return dataset.dataframe.select([0])


class DataGenDataset(RLHFDataset):
    """
    A dataset class that uses a data generation strategy to process data.
    This class extends RLHFDataset and uses an AbstractDataGen instance to generate data.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        super().__init__(data_files, tokenizer, config, processor)
        self.datagen: AbstractDataGen = config.datagen
        assert "datagen" in config and config.datagen.get("path", None) is not None, (
            f"datagen path is not set in config: {config}"
        )
        # Dynamically load the custom datagen class
        datagen_cls = load_extern_type(config.datagen.path, config.datagen.name)

        # Verify that the custom datagen class inherits from AbstractDataGen
        abs_cls = AbstractDataGen
        if not issubclass(datagen_cls, abs_cls):
            raise TypeError(
                f"The custom datagen class '{config.datagen.name}' from '{config.datagen.path}'"
                + " must inherit from {abs_cls}"
            )

        self.data_generator = datagen_cls(config.datagen)
        self.on_epoch_end()

    def append_dataframe(self, new_dataframe: datasets.Dataset):
        new_dataframe = self.maybe_filter_out_long_prompts(new_dataframe)
        self.dataframe = datasets.concatenate_datasets([self.dataframe, new_dataframe])

        print(f"new dataset len: {len(self.dataframe)}")

    def on_epoch_end(self) -> None:
        """
        Generate data using the provided data generation strategy.
        """
        d = self.data_generator.generate(self)
        self.append_dataframe(d)
