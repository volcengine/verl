# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from abc import ABC, abstractmethod
from typing import Generator, TypedDict

import torch


class TensorMeta(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


class CheckpointEngineRegistry:
    """Checkpoint engine registry."""

    _registry: dict[str, type["CheckpointEngine"]] = {}

    def register(backend: str):
        """Register a checkpoint engine.

        Args:
            backend: The backend of the checkpoint engine.
        """

        def wrapper(cls: type["CheckpointEngine"]):
            CheckpointEngineRegistry._registry[backend] = cls
            return cls

        return wrapper

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        """Create a new checkpoint engine instance.

        Args:
            backend: The backend of the checkpoint engine.
            *args: Variable length argument pass to the checkpoint engine constructor.
            **kwargs: Arbitrary keyword arguments pass to the checkpoint engine constructor.

        Returns:
            A new checkpoint engine instance.
        """
        if backend not in cls._registry:
            raise ValueError(f"Checkpoint engine {backend} not registered")
        return cls._registry[backend](*args, **kwargs)


class CheckpointEngine(ABC):
    """CheckpointEngine is an abstraction to transfer weights from trainer to rollout.

    In trainer process:
    >>> trainer = EngineRegistry.new(...) # FSDP, Megatron, VeOmini, TorchTitan, ...
    >>> engine = CheckpointEngine.new(...) # NCCLCheckpointEngine, NIXLCheckpointEngine, ...
    >>> await engine.send_weights(trainer.get_per_tensor_param())

    In rollout process:
    >>> engine = CheckpointEngine.new(...)
    >>> server_adapter = ServerAdapter()
    >>> await server_adapter.update_weights(engine.get_weights()) # update weights via cuda ipc
    """

    @abstractmethod
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


class CheckpointEngineWithCache(CheckpointEngine):
    """Checkpoint engine with local cache: shm, disk, etc. This allow to synchronize weights without interrupting
    rollout ongoing requests (partial rollout). After requests exhausted, rollout can get weights from local cache.

    Laminar: https://arxiv.org/abs/2510.12633
    """

    @abstractmethod
    async def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get the weights of the model from local cache.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


@CheckpointEngineRegistry.register("naive")
class ColocatedCheckpointEngine(CheckpointEngine):
    """Checkpoint engine for trainer and rollout colocated on same GPU.

    In trainer process:
    >>> engine = ColocatedCheckpointEngine()
    >>> trainer = Trainer()
    >>> server_adapter = ServerAdapter()
    >>> engine.send_weights(trainer.get_per_tensor_param())
    >>> server_adapter.update_weights(engine.receive_weights())
    """

    def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None]):
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        self.weights = weights

    def receive_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        yield from self.weights
        self.weights = None
