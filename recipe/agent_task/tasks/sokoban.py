from copy import deepcopy

from recipe.agent_task.agents.sokoban_agent import SokobanAgent
from verl.task.interface import TaskInterface


class Sokoban(TaskInterface):
    def __init__(self, item, agent_access, episode_num=1, max_steps=100):
        super().__init__()
        import gym
        import gym_sokoban

        print(gym_sokoban.env_json)
        self.env = gym.make("Sokoban-v0")
        self.action_lookup = self.env.unwrapped.get_action_lookup()

        self.agent = SokobanAgent(access=agent_access, action_lookup=self.action_lookup)

        self.episode_num = episode_num
        self.max_steps = max_steps

    def run(self):
        for episode_idx in range(self.episode_num):
            observation = self.env.reset()
            for step in range(self.max_steps):
                action = self.agent(img_ndarray=observation)
                old_observation = deepcopy(observation)
                observation, reward, done, info = self.env.step(action)
                self._trajectory.append({"observation": old_observation, "action": action, "reward": reward, "done": done, "info": info})
                if done:
                    break
        self.env.close()
