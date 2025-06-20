from verl.task.interface import AgentInterface
from verl.workers.rollout.client import OpenAIClient


class VanillaAgent(AgentInterface):
    """
    A vanilla agent only convert input prompt to a message (OpenAI API) and post request.
    """

    def __init__(self, access):
        self.client = OpenAIClient(rollout_access=access)

    def _preprocess_prompt(self, prompt):
        messages = []
        return messages

    def __call__(self, *, prompt, **kwargs):
        messages = self._preprocess_prompt(prompt=prompt)
        return self.client.response.create(model=None, messages=messages, **kwargs)
