import time

from recipe.agent_task.agents.calculator_agent import AgentWithCalculator
from verl.task.interface import TaskInterface


class GSM8K(TaskInterface):
    def __init__(self, item, agent_access):
        self.agent = AgentWithCalculator(access=agent_access, max_round_per_prompt=5)
        self.system_prompt = item["prompt"][0]["content"]
        self.prompt = item["prompt"][1]["content"]
        self.ground_truth = item["reward_model"]["ground_truth"]
        self._trajectory = []

    def run(self):
        st = time.time()
        messages = self.agent(prompt=self.prompt, system_prompt=self.system_prompt)
        et = time.time()
        final_response = messages[-1]
        assert final_response["role"] == "agent"
        reward = self.ground_truth == final_response["content"]

        self._trajectory.append({"messages": messages, "reward": reward, "done": True, "timing": et - st})
