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
"""
Create dataset for calculator
"""

import random

import pandas as pd


def generate_math_expression(min_terms=2, max_terms=5, min_number=1, max_number=10, allow_decimals=False, max_depth=2):
    """
    Generate a random mathematical expression with operators +, -, *, /, and parentheses.

    Args:
        min_terms (int): Minimum number of terms in the expression.
        max_terms (int): Maximum number of terms in the expression.
        max_number (int): Maximum value for numbers in the expression.
        allow_decimals (bool): Whether to allow decimal numbers.
        max_depth (int): Maximum nesting depth for parentheses.

    Returns:
        str: A valid mathematical expression as a string.
    """

    def generate_number():
        """Generate a random number (integer or float)."""
        assert min_number < max_number
        num = random.uniform(min_number, max_number)
        if not allow_decimals:
            num = int(num)
        else:
            num = round(num, random.randint(0, 2))  # Round to 0-2 decimal places
        return str(num)

    def generate_term(depth=0):
        """Generate a term (number or parenthesized expression)."""
        if depth < max_depth and random.random() < 0.5:  # 50% chance to add parentheses
            expr = generate_expression(depth + 1)
            return f"({expr})"
        else:
            return generate_number()

    def generate_expression(depth=0):
        """Generate a full expression with multiple terms and operators."""
        num_terms = random.randint(min_terms, max_terms)
        terms = [generate_term(depth) for _ in range(num_terms)]

        # Randomly select operators
        operators = ["+", "-", "*", "/", "@"]
        expr = terms[0]

        for i in range(1, num_terms):
            # Bias towards + and - for readability
            op = random.choices(
                operators,
                weights=[0, 0, 0, 0, 1],  # + and - are 1.5x more likely than * and /
            )[0]
            expr += f" {op} " + terms[i]

        return expr

    return generate_expression()


def test():
    # Example 1: Basic integer expression
    print(generate_math_expression())
    # Output: (3 + 7) * 2 - 5

    # Example 2: Expression with decimals
    print(generate_math_expression(allow_decimals=True))
    # Output: 4.5 / (2.1 + 3.7) - 1.2

    # Example 3: More complex expression with higher depth
    print(generate_math_expression(max_terms=6, max_depth=3))
    # Output: ((5 * 2) - (3 + 1)) / (7 - 2) + 4

    # Example 4: Simplified expression
    print(generate_math_expression(min_terms=2, max_terms=3, max_number=5))
    # Output: 4 - 2 * 3


def calculate(expression: str) -> float:
    """
    Evaluate a mathematical expression with +, -, *, /, @, and parentheses.
    The @ operator is defined as: a @ b = 3a - 2b.

    Args:
        expression (str): Input mathematical expression (e.g., "3@2+4").

    Returns:
        float: Result of the evaluated expression.

    Raises:
        ValueError: For invalid expressions (e.g., mismatched parentheses, division by zero).
    """

    def tokenize(s: str) -> list:
        """Convert the input string into tokens (numbers, operators, parentheses)."""
        tokens = []
        i = 0
        while i < len(s):
            if s[i].isdigit() or s[i] == ".":
                # Parse number (integer or float)
                j = i
                while j < len(s) and (s[j].isdigit() or s[j] == "."):
                    j += 1
                tokens.append(s[i:j])
                i = j
            elif s[i] in "+-*/@()":
                # Operator or parenthesis
                tokens.append(s[i])
                i += 1
            elif s[i].isspace():
                # Skip whitespace
                i += 1
            else:
                raise ValueError(f"Invalid character: {s[i]}")
        return tokens

    def infix_to_postfix(tokens: list) -> list:
        """Convert infix notation to postfix notation (Reverse Polish Notation)."""
        output = []
        stack = []
        # Higher precedence for @ (between * and +)
        precedence = {"@": 3, "*": 2, "/": 2, "+": 1, "-": 1}

        for token in tokens:
            if token.isdigit() or "." in token:
                output.append(token)
            elif token == "(":
                stack.append(token)
            elif token == ")":
                while stack and stack[-1] != "(":
                    output.append(stack.pop())
                if not stack or stack[-1] != "(":
                    raise ValueError("Mismatched parentheses")
                stack.pop()  # Discard '('
            else:  # Operator
                while stack and stack[-1] != "(" and precedence.get(stack[-1], 0) >= precedence.get(token, 0):
                    output.append(stack.pop())
                stack.append(token)

        # Pop remaining operators
        while stack:
            if stack[-1] in "()":
                raise ValueError("Mismatched parentheses")
            output.append(stack.pop())

        return output

    def evaluate_postfix(postfix: list) -> float:
        """Evaluate postfix expression using a stack."""
        stack = []
        for token in postfix:
            if token.isdigit() or "." in token:
                stack.append(float(token))
            else:
                if len(stack) < 2:
                    raise ValueError("Invalid expression")
                b = stack.pop()
                a = stack.pop()
                if token == "+":
                    res = a + b
                elif token == "-":
                    res = a - b
                elif token == "*":
                    res = a * b
                elif token == "/":
                    if b == 0:
                        raise ValueError("Division by zero")
                    res = a / b
                elif token == "@":
                    res = 3 * a - 2 * b  # Custom @ operator implementation
                else:
                    raise ValueError(f"Invalid operator: {token}")
                stack.append(res)

        if len(stack) != 1:
            raise ValueError("Invalid expression")
        return stack[0]

    # Remove spaces and validate parentheses
    expression = expression.replace(" ", "")
    if expression.count("(") != expression.count(")"):
        raise ValueError("Mismatched parentheses")

    tokens = tokenize(expression)
    postfix = infix_to_postfix(tokens)
    result = evaluate_postfix(postfix)

    # Convert integers to integer representation
    if result.is_integer():
        return int(result)
    return result


def generate_data(total_num_dataset, split):
    rl_dataset = {
        "prompt": [],
        "data_source": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
        "agent_name": [],
    }

    for idx in range(total_num_dataset):
        while True:
            try:
                expression: str = generate_math_expression(
                    min_terms=2, max_terms=3, min_number=1, max_number=10, allow_decimals=False, max_depth=1
                )

                num_plus = expression.count("+")
                num_minus = expression.count("-")
                num_mul = expression.count("*")
                num_star = expression.count("@")

                answer = str(calculate(expression))
                # answer = str(eval(expression))
                break
            except Exception as e:
                print(e)
                continue

        num_tool_calls = num_plus + num_minus + num_mul + num_star

        prompt = (
            f"We define a new math operator @, where you can only call an external tool to compute. "
            f"Please put your final answer inside \\boxed{{}} only in the last turn. Now answer the "
            f"following questions:\nCompute {expression}"
        )
        prompt_with_template = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        rl_dataset["prompt"].append(prompt_with_template)
        rl_dataset["data_source"].append("lighteval/MATH")
        rl_dataset["ability"].append("math")
        rl_dataset["reward_model"].append({"style": "lighteval/MATH", "ground_truth": answer})
        rl_dataset["extra_info"].append(
            {"index": idx, "expression": expression, "split": split, "expected_tool_calls": num_tool_calls}
        )
        rl_dataset["agent_name"].append("math_expression")

    rl_dataset = pd.DataFrame(data=rl_dataset)
    return rl_dataset


if __name__ == "__main__":
    # print(calculate("3@2"))          # Output: 5 (3*3 - 2*2)
    # print(calculate("3@2+4"))        # Output: 9 (5 + 4)
    # print(calculate("3*(4@2)"))      # Output: 24 (3 * 8)
    # print(calculate("(5@3)*2"))      # Output: 18 (9 * 2)

    train_dataset = generate_data(total_num_dataset=5000, split="train")
    test_dataset = generate_data(total_num_dataset=500, split="test")

    train_dataset.to_parquet("train.parquet")
    test_dataset.to_parquet("test.parquet")
