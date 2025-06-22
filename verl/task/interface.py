from abc import ABC, abstractmethod


class TaskInterface(ABC):
    """
    Task represents a RLHF task.
    The main purpose of this interface is not force the API but reveal the concept of Task Loop,
    a top-down paradigm to write the process of a task, rather than code it recursively through Callbacks.
    """

    def __init__(self):
        self._trajectory = []

    @abstractmethod
    def run(self):
        # run Task Loop here
        ...

    @property
    def trajectory(self):
        return self._trajectory


class AgentInterface(ABC):
    """
    Agent represents a LLM (client) integrated with tools.
    However, it is not a must to embed tools to Agents. Tools can also be called in the Task Loop.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs): ...
