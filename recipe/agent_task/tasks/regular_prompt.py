from collections import namedtuple

from verl.task.interface import TaskInterface
from verl.workers.rollout.client import OpenAIClient

StepSample = namedtuple("StepSample", ["observation", "action", "reward", "done"])


class RegularPrompt(TaskInterface):
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        self._trajectory = []

    def _preprocess_prompt(self, prompt):
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        return messages

    def run(self, *args, origin_prompt, client: OpenAIClient, **kwargs):
        messages = self._preprocess_prompt(origin_prompt)

        response = client.response.create(model="", messages=messages)

        self._trajectory.append(StepSample(observation=messages, action=response.choices[0].message.content, reward=None, done=True))
