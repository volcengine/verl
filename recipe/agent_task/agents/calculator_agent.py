import logging
from copy import deepcopy

from verl.task.interface import AgentInterface
from verl.workers.rollout.client import OpenAIClient


class Calculator:
    @classmethod
    def run(cls, math_expr):
        try:
            assert isinstance(math_expr, str), "content must be string"
            result = eval(math_expr)
        except Exception as e:
            logging.warning(f"failed to calculate {math_expr}: {e}")
        else:
            return result


class AgentWithCalculator(AgentInterface):
    def __init__(self, access, max_round_per_prompt=5):
        super().__init__()
        self.client = OpenAIClient(rollout_access=access)
        self.messages = []
        self.max_round_per_prompt = max_round_per_prompt

    def extract_math_expression(self, content) -> str: ...

    def response_completed(self, content) -> bool: ...

    def __call__(self, *, prompt, system_prompt=None, **kwargs):
        if system_prompt is None:
            system_prompt = "You are a useful agent. You can use calculator to eval math expression " + 'by wrapping the expression with pattern "<Calculate>{math_expression}</Calculator>"'
        self.messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        round_count = 0
        while round_count < self.max_round_per_prompt:
            response = self.client.response.create(model=None, messages=self.messages, **kwargs)
            content = response.choices[0].content
            self.messages.append({"role": "agent", "content": deepcopy(content)})

            if self.response_completed(content):
                return self.messages

            math_expression = self.extract_math_expression(content=content)
            if math_expression is not None:
                math_result = Calculator.run(math_expression)
                self.messages.append({"role": "calculator", "content": f"{math_expression}={math_result}"})

            round_count += 1

        return self.messages
