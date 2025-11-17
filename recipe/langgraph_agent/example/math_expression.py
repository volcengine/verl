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
import re
import json
from langchain_core.tools import tool
from recipe.langgraph_agent.react_agent_loop import ReactAgentLoop

def safe_int(x):
    """Safely convert to int, return None if not a valid number\n
    Due to the maximum response length limit, the LLM may not strictly respond in JSON format, causing errors in the model tools' operations."""
    try:
        return int(x)
    except:
        return None
def _eval_expr(expression: str):
    """
    Safely evaluate expression: Even if LLM gives junk input, it won't crash.
    """
    if not expression or not isinstance(expression, str):
        return 0
    expr = expression.strip()
    if expr == "":
        return 0

    # Fix handling of negative numbers
    if expr.startswith("-"):
        expr = "0 " + expr

    # Recursively handle parentheses
    while "(" in expr:
        expr = re.sub(
            r"\(([^()]+)\)",
            lambda m: str(_eval_expr(m.group(1))),
            expr
        )
    # Clean invalid characters 
    expr = re.sub(r"[^0-9+\-*@\s]", " ", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    if expr == "":
        return 0
    # Split tokens
    tokens = re.split(r"([+\-*@])", expr)
    tokens = [t.strip() for t in tokens if t.strip() != ""]
    # Merge negative signs
    merged = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "-" and (i == 0 or tokens[i - 1] in "+-*@") and i + 1 < len(tokens):
            if re.fullmatch(r"-?\d+", tokens[i + 1]):
                merged.append(str(-int(tokens[i + 1])))
                i += 2
                continue
        merged.append(tokens[i])
        i += 1
    tokens = merged
    # Filter valid tokens
    valid_tokens = [t for t in tokens if re.fullmatch(r"-?\d+", t) or t in "+-*@"]
    if len(valid_tokens) == 0:
        # If no valid tokens, return 0
        return 0
    if len(valid_tokens) == 1:
        v = safe_int(valid_tokens[0])
        return v if v is not None else 0

    def calc(a, op, b):
        a, b = safe_int(a), safe_int(b)
        if a is None or b is None:
            return 0

        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "@": return 3 * a - 2 * b
        return 0

    res = safe_int(valid_tokens[0]) or 0
    i = 1
    while i < len(valid_tokens) - 1:
        op = valid_tokens[i]
        b = safe_int(valid_tokens[i + 1])
        if b is None:
            i += 2
            continue
        res = calc(res, op, b)
        i += 2

    return res


@tool(parse_docstring=True)
def calculate(expression: str) -> int:
    """
     Evaluate a full mathematical expression with custom '@' operator.

     Args:
         expression: A string expression using '+', '-', '*', and '@'.
             '@' means (3*a - 2*b). Parentheses are supported.

     Examples:
         >>> calculate("3 @ (9 @ 4 @ 4) @ (2 @ 2 @ 2)")
     """
    print(f"Received expression: {expression}")
    # LangGraph tool sometimes passes dict
    if isinstance(expression, dict):
        expression = expression.get("expression", "")

    # Try to parse JSON
    try:
        obj = json.loads(expression)
        if "expression" in obj:
            expression = obj["expression"]
    except:
        # Fix some broken JSON formats
        s = expression.replace("expression:", '"expression":')
        s = s.replace("'", '"')
        try:
            obj = json.loads(s)
            expression = obj.get("expression", expression)
        except:
            pass

    try:
        return _eval_expr(expression)
    except:
        return 0


class MathExpressionReactAgentLoop(ReactAgentLoop):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        cls.tools = [calculate]
        super().init_class(config, tokenizer)


__all__ = ["calculate", "MathExpressionReactAgentLoop"]