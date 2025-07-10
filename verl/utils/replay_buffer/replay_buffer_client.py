from abc import ABC, abstractmethod
from typing import Any, List, Union

from verl import DataProto


class ReplayBufferClient(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def push(self, key: str, batches: Union[Any, List[DataProto]]) -> None:
        pass

    @abstractmethod
    def get(self, key: str):
        pass

    @abstractmethod
    def delete(self, key: str):
        pass

    @abstractmethod
    def sample(self):
        pass
