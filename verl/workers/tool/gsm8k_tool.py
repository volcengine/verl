import json
from typing import Optional, Tuple
from uuid import uuid4
from .base_tool import BaseTool
from .data_model import OpenAIFunctionToolSchema, OpenAIFunctionParametersSchema, OpenAIFunctionParsedSchema
from verl.utils.reward_score import gsm8k


class Gsm8kTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "calc_gsm8k_reward",
                "description": "A tool for calculating the reward of gsm8k",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The answer to the question",
                        },
                    },
                    "required": ["answer"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema
    
    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id
    
    async def execute(self, instance_id: str, parameters: str, **kwargs) -> Tuple[str, float, dict]:
        try:
            _parameters = json.loads(parameters)
        except json.JSONDecodeError as e:
            print(f"Failed to parse parameters: {parameters}, JSONDecodeError: {e}")
            _parameters = {}
        print(f"{_parameters=}")
        if isinstance(_parameters, dict):
            answer = _parameters.get("answer", "")
            if not isinstance(answer, str):
                answer = str(answer)
        else:
            answer = ""
        self._instance_dict[instance_id]["response"] = "#### " + answer
        reward = await self.calc_reward(instance_id)
        # penalty for non improved answer submission
        tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
        # update the reward
        self._instance_dict[instance_id]["reward"] = reward
        return f"Current answer {reward=}", tool_reward, {}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return gsm8k.compute_score(self._instance_dict[instance_id]["response"], self._instance_dict[instance_id]["ground_truth"])
    
    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
