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
    except (ValueError, TypeError):
        return None

def _eval_expr(expression: str):
    """
    Safely evaluate expression with operator precedence using RPN (Reverse Polish Notation).
    Operators:
      +, -, *, @
    Precedence:
      * and @  >  + and -
    All operators are left-associative.
    Unary minus is supported (e.g., -5, 3 + -2, (-3) * 4).
    """
    # Basic type / empty checks
    if not expression or not isinstance(expression, str):
        return 0

    expr = expression.strip()
    if expr == "":
        return 0

    # "-5 * 2" -> "0 -5 * 2"
    if expr.startswith("-"):
        expr = "0 " + expr

    # Clean invalid characters but keep parentheses
    expr = re.sub(r"[^0-9+\-*@()\s]", " ", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    if expr == "":
        return 0

    #Tokenize: numbers, operators, parentheses
    raw_tokens = re.findall(r"\d+|[@+\-*()]", expr)
    if not raw_tokens:
        return 0

    tokens = []
    i = 0
    while i < len(raw_tokens):
        t = raw_tokens[i]
        if (
            t == "-"
            and i + 1 < len(raw_tokens)
            and re.fullmatch(r"\d+", raw_tokens[i + 1])
            and (i == 0 or raw_tokens[i - 1] in "+-*@(")
        ):
            tokens.append("-" + raw_tokens[i + 1])
            i += 2
            continue
        tokens.append(t)
        i += 1

    filtered = []
    for t in tokens:
        if re.fullmatch(r"-?\d+", t) or t in "+-*@()":
            filtered.append(t)
    tokens = filtered

    if not tokens:
        return 0
   # Single Digit Calculation
    if len(tokens) == 1 and re.fullmatch(r"-?\d+", tokens[0]):
        v = safe_int(tokens[0])
        return v if v is not None else 0

   
    precedence = {"+": 1, "-": 1, "*": 2, "@": 2} # RPN:Priority definition: * and @ highest, + and - next, all left-associative

    output = []      # RPN output queue
    op_stack = []    # Operator Stack

    for t in tokens:
        if re.fullmatch(r"-?\d+", t):
            output.append(t)
        elif t in precedence:
            while (
                op_stack
                and op_stack[-1] in precedence
                and precedence[op_stack[-1]] >= precedence[t]
            ):
                output.append(op_stack.pop())
            op_stack.append(t)
        elif t == "(":
            op_stack.append(t)
        elif t == ")":
            while op_stack and op_stack[-1] != "(":
                output.append(op_stack.pop())
            if op_stack and op_stack[-1] == "(":
                op_stack.pop()

    while op_stack:
        op = op_stack.pop()
        if op in precedence:
            output.append(op)

    if not output:
        return 0

    def calc(a, op, b):
        a, b = safe_int(a), safe_int(b)
        if a is None or b is None:
            return 0

        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "@":
            return 3 * a - 2 * b
        return 0

    stack = []
    for tok in output:
        if re.fullmatch(r"-?\d+", tok):
            stack.append(tok)
        elif tok in precedence:
            if len(stack) < 2:
                return 0
            b = stack.pop()
            a = stack.pop()
            result = calc(a, tok, b)
            stack.append(str(result))

    if not stack:
        return 0

    final = safe_int(stack[-1])
    return final if final is not None else 0



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
    except Exception as e:
        print(f"eorrs:{e}")
        return 0


class MathExpressionReactAgentLoop(ReactAgentLoop):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        cls.tools = [calculate]
        super().init_class(config, tokenizer)


__all__ = ["calculate", "MathExpressionReactAgentLoop"]