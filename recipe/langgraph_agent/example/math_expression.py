# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from langchain_core.tools import tool

from recipe.langgraph_agent.react_agent_loop import ReactAgentLoop


@tool(parse_docstring=True)
def calculate(a: int, b: int, operand: str) -> int:
    """
    Compute the results using operand with two integers

    Args:
        a: the first operand
        b: the second operand
        operand: '+' or '-' or '*' or '@'
    """
    assert operand in ["+", "-", "*", "@"], f"unknown operand {operand}"
    if operand == "@":
        return 3 * a - 2 * b
    return eval(f"{a} {operand} {b}")


class MathExpressionReactAgentLoop(ReactAgentLoop):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        cls.tools = [calculate]
        super().init_class(config, tokenizer)
