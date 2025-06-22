import base64
import io
from typing import Dict

from PIL import Image

from verl.task.interface import AgentInterface
from verl.workers.rollout.client import OpenAIClient


class SokobanAgent(AgentInterface):
    """
    A sokoban agent can play sokoban game.
    """

    def __init__(self, access, action_lookup: Dict, bon=1):
        super().__init__()
        self.client = OpenAIClient(rollout_access=access)
        self.action_lookup = action_lookup
        self.system_prompt = {"role": "system", "content": f"You are Sokoban player. You have the following actions available to take: {', '.join(list(action_lookup.values()))}."}
        self.user_prompt_content = "Now, the game snapshot becomes this image. Which action should I take?"
        self.bon = bon

    def _content_to_action(self, content):
        for action, action_name in self.action_lookup.items():
            if action_name in content:  # for sokoban, there is no overlapping in action_names
                return action

    def __call__(self, *, img_ndarray, **kwargs):
        image = Image.fromarray(img_ndarray, mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        base64_bytes = base64.b64encode(buffer.read()).decode("utf-8")

        messages = [self.system_prompt, {"role": "user", "content": [{"type": "text", "text": "Describe this image."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_bytes}", "detail": "high"}}]}]

        response = self.client.response.create(model="", n=self.bon, messages=messages, **kwargs)

        actions = [self._content_to_action(choice.content) for choice in response.choices]

        return actions
