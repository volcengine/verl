from abc import ABC, abstractmethod


class ReplayBufferInterface(ABC):
    @abstractmethod
    def put(self, *args, **kwargs): ...

    @abstractmethod
    def get(self, *args, **kwargs): ...
