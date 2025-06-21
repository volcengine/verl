from verl.task.interface import TaskInterface
from verl.workers.rollout.client import OpenAIClient


class RegularPrompt(TaskInterface):
    def __init__(self, item, agent_access):
        self.system_prompt = ""
        self._trajectory = []
        self.origin_prompt = item["prompt"][0]
        self.client = OpenAIClient(llmserver_manager=agent_access)

    def _preprocess_prompt(self, prompt):
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        return messages

    def run(self):
        messages = self._preprocess_prompt(self.origin_prompt)

        response = self.client.response.create(model="", messages=messages)

        self._trajectory.append(dict(observation=messages, action=response.choices[0].message.content, reward=None, done=True))
