# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
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

import math
import os
import re
import string
from typing import Any

import sympy
from agentlightning import LLM, LitAgent, NamedResources, Trainer, configure_logger, reward
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

configure_logger()

calculator_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-calculator"])


# Copied and adapted from https://github.com/prompteus/calc-x/blob/master/gadgets/metrics.py


def normalize_option(option: str) -> str:
    """
    >>> normalize_option("  (A)  \n")
    'A'
    """
    return re.sub(r"(\s+|\(|\))", "", option)


def is_option_result(result: str) -> bool:
    """
    >>> is_option_result("  A)  \n")
    True
    >>> is_option_result("  23/7 ")
    False
    """
    return normalize_option(result) in list(string.ascii_letters)


def float_eval(input_str: str) -> float:
    if " = around " in input_str:
        input_str = input_str.split(" = around ")[0]
    expr = sympy.parse_expr(input_str, evaluate=True)
    return float(expr.evalf())


def scalar_are_results_same(pred_result: str, true_result: str, rel_tol: float) -> bool:
    pred_result = str(pred_result) if pred_result is not None else ""
    true_result = str(true_result) if true_result is not None else ""

    if pred_result.strip() == true_result.strip():
        return True

    if is_option_result(true_result):
        # The task is to select correct option
        true_result = normalize_option(true_result)
        pred_result = normalize_option(pred_result)
        return pred_result == true_result

    # The task is to calculate the result as a number
    try:
        pred_float = float_eval(pred_result)
        true_float = float_eval(true_result)
        return math.isclose(pred_float, true_float, rel_tol=rel_tol)
    except Exception:
        pass

    return False


@reward
async def eval(prediction: str, ground_truth: str) -> float:
    return float(scalar_are_results_same(prediction, ground_truth, 1e-2))


def get_agent(model, openai_base_url, temperature, workbench):
    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=openai_base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        temperature=temperature,
    )

    calc_agent = AssistantAgent(
        name="calc",
        model_client=model_client,
        workbench=workbench,
        reflect_on_tool_use=True,
    )
    return calc_agent


class CalcAgent(LitAgent):
    async def training_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        llm: LLM = resources.get("main_llm")
        async with McpWorkbench(calculator_mcp_server) as workbench:
            calc_agent = get_agent(
                llm.model,
                llm.endpoint,
                llm.sampling_parameters.get("temperature", 0.7),
                workbench,
            )
            try:
                output_format = (
                    "Output the answer when you are ready. "
                    "The answer should be surrounded by three sharps (`###`), "
                    "in the form of ### ANSWER: <answer> ###."
                )
                prompt = task["question"] + " " + output_format
                result = await calc_agent.run(task=prompt)
                # evaluate
                answer = re.search(r"###\s*ANSWER:\s*(.+?)(\s*###|$)", result.messages[-1].content)
                if answer:
                    answer = answer.group(1)
                else:
                    answer = result.messages[-1].content
            except Exception as e:
                print("Failure:", str(e))
                answer = "None"
            reward = await eval(answer, str(task["result"]))  # reward is tracked with the decorator
            print("answer: {} ground_truth: {} reward: {}".format(answer, task["result"], reward))

    async def validation_rollout_async(self, task: Any, rollout_id: str, resources: NamedResources) -> Any:
        llm: LLM = resources.get("main_llm")
        resources = {
            "main_llm": LLM(
                endpoint=llm.endpoint,
                model=llm.model,
                sampling_parameters={"temperature": 0},
            )
        }
        return await self.training_rollout_async(task, rollout_id, resources)


if __name__ == "__main__":
    Trainer(n_workers=10).fit(CalcAgent(), "http://localhost:9999/")
