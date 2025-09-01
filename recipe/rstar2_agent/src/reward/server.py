# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Any

import aiohttp
import asyncio
import base64
import re
import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

from recipe.rstar2_agent.src.tools.code_judge_utils import run_tool_calls_on_server_async

verify_math_prefix = """
from recipe.rstar2_agent.src.reward.compute_score import compute_score
import base64
solution_str = base64.b64decode("{}".encode()).decode()
ground_truth = base64.b64decode("{}".encode()).decode()
result = compute_score(solution_str, ground_truth)
"""

verify_math_suffix = """
print(f"<result>{result}</result>")
"""


@register("code_judge")
class CodeJudgeRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the CodeJudgeRewardManager instance.

        Note that num_examine, compute_score, reward_fn_key is not used in this implementation.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        for i in range(0, len(data), 64):
            batch_data = data[i : i + 64]
            tool_calls = []
            for j in range(len(batch_data)):
                data_item = batch_data[j]  # DataProtoItem

                if "response_text" in data_item.non_tensor_batch and data_item.non_tensor_batch["response_text"] is not None:
                    response_str = data_item.non_tensor_batch["response_text"]
                else:
                    response_ids = data_item.batch["responses"]
                    prompt_length = data_item.batch["prompts"].shape[-1]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_ids = response_ids[:valid_response_length]
                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
                tool_calls.append(self.create_tool_call(response_str, ground_truth))

            results = self.execute_tool_calls(tool_calls)
            for j in range(len(results)):
                reward_tensor[i * 64 + j, valid_response_length - 1] = results[j]

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def create_tool_call(self, solution_str: str, ground_truth: str):
        ground_truth = str(ground_truth)
        a = base64.b64encode(solution_str.encode()).decode()
        b = base64.b64encode(ground_truth.encode()).decode()
        code = verify_math_prefix.format(a, b) + verify_math_suffix
        return {
            "name": "compute_score",
            "arguments": {
                "code": code
            }
        }

    def extract_tool_call_result(self, result: str):
        if result is None:
            return 0.0
        match = re.search(r'<result>(.*?)</result>', result)
        return float(match.group(1)) if match else 0.0

    def execute_tool_calls(self, tool_calls):
        async def run_tool_calls(tool_calls):
            tool_connector = aiohttp.TCPConnector(limit=32, force_close=True, enable_cleanup_closed=True)
            tool_timeout = aiohttp.ClientTimeout(total=60)
            tool_session = aiohttp.ClientSession(connector=tool_connector, timeout=tool_timeout)
            responses = await run_tool_calls_on_server_async(
                tool_calls=tool_calls,
                session=tool_session,
                generate_tool_call_code=lambda x: x["arguments"]["code"],
                generate_tool_call_input=lambda x: None,
            )
            await tool_session.close()
            return responses

        results = asyncio.run(run_tool_calls(tool_calls))
        return [self.extract_tool_call_result(result) for result in results]
