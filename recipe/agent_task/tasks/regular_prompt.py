from collections import namedtuple

from verl.task.interface import TaskInterface
from verl.workers.rollout.client import OpenAIClient

StepSample = namedtuple("StepSample", ["observation", "action", "reward", "done"])


class RegularPrompt(TaskInterface):
    def __init__(self, system_prompt, origin_prompt, client: OpenAIClient):
        self.system_prompt = system_prompt
        self._trajectory = []
        self.origin_prompt = origin_prompt
        self.client = client

    def _preprocess_prompt(self, prompt):
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        return messages

    def run(self):
        messages = self._preprocess_prompt(self.origin_prompt)

        response = self.client.response.create(model="", messages=messages)

        self._trajectory.append(StepSample(observation=messages, action=response.choices[0].message.content, reward=None, done=True))
