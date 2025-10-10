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
import logging
import re
from typing import Any

import datasets
from pathlib import Path

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # sandbox has no score or metrics, use Nones
        return result, None, None



class SumbitTool:
    def __init__(self, config, tool_schema: OpenAIFunctionToolSchema):
        self.name="submit"
        self.tool_schema = tool_schema
        self.config = config

    async def execute(self, *args, **kwargs):
        raise NotImplementedError



class CustomRLHFDataset(RLHFDataset):
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset(parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")
        save_key = "fakearc"
        save_path = Path("~/Documents/arc-agi/data/misc").expanduser() / f"data_{save_key}.pq"
        self.dataframe.to_parquet(save_path)
        print("="*200+f"\nSAVED DATASET TO {save_path}\n"+"="*200)



def compute_score(data_source, solution_str, ground_truth, extra_info):
    print(f"SOLUTION_STR_IN_COMPUTE_SCORE:\n {solution_str}\nEND SOLUTION_STR_IN_COMPUTE_SCORE")
    print(f"EXTRA_INFO_IN_COMPUTE_SCORE:\n {extra_info}\nEND EXTRA_INFO_IN_COMPUTE_SCORE")
#    # use \\boxed{...} answer
#    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
#
#    # encourage model to call tools
#    num_turns = extra_info["num_turns"]
#    if num_turns is None:
#        num_turns = 1
#    if result["score"] < 0:
#        tool_call_reward = (num_turns - 2) / 2 * 0.1
#        result["score"] = min(-0.6, result["score"] + tool_call_reward)
#
#    if result["pred"] is None:
#        result["pred"] = ""
    return 0.0
