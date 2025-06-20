from collections import namedtuple

from verl.workers.rollout.async_server import AsyncLLMServerManager

Response = namedtuple("Response", ["create"])


class OpenAIClient:
    """
    A mocked OpenAI Python Client, using the `ChatScheduler` as the server de facto.
    The client could be used either on driver process or remote ray actors. For the
    latter scenario, the `ChatScheduler` should also be a ray actor.
    """

    def __init__(self, *args, llmserver_manager: AsyncLLMServerManager, **kwargs):
        self.chat_scheduler = llmserver_manager.chat_scheduler
        self.remote_scheduler = llmserver_manager.remote_scheduler

        self._response = Response(self._create)

    def _create(self, *, model=None, messages=None, **kwargs):
        assert messages is not None
        # TODO(zw0601): implement here

    @property
    def response(self):
        return self._response
