# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modifications Copyright 2025 SPO authors
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

import datasets

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    @rollout_trace_op
    async def execute(self, instance_id: str, code: str, **kwargs) -> tuple[str, float, dict]:
        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        timeout = self.default_timeout
        language = self.default_language
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # sandbox has no score or metrics, use Nones
        return result, None, None


answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            if ".parquet" in parquet_file:
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            elif "open-r1/DAPO-Math-17k-Processed" in parquet_file:
                dataframe = datasets.load_dataset(parquet_file, "all")["train"]
            elif "ByteDance-Seed/BeyondAIME" in parquet_file:
                dataframe = datasets.load_dataset(parquet_file)["test"]
            elif "Polaris-Dataset-Hard" in parquet_file:
                dataframe = datasets.load_from_disk(parquet_file)
            else:
                dataframe = datasets.load_dataset(parquet_file)["train"]
            data_source = "/".join(parquet_file.split("/")[-2:])
            if data_source in [
                "Maxwell-Jia/AIME_2024",
                "yentinglin/aime_2025",
                "ByteDance-Seed/BeyondAIME",
                "MathArena/brumo_2025",
                "MathArena/hmmt_feb_2025",
            ]:
                dataframe = dataframe.map(
                    self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                )
            elif "Polaris-Dataset-Hard" in data_source:
                dataframe = dataframe.map(
                    self.map_fn,
                    fn_kwargs={"data_source": "dataset/Polaris-Dataset-Hard"},
                    remove_columns=dataframe.column_names,
                )
            else:
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        if data_source == "Maxwell-Jia/AIME_2024":
            problem, answer = row["Problem"], row["Answer"]
        elif data_source in [
            "yentinglin/aime_2025",
            "ByteDance-Seed/BeyondAIME",
            "MathArena/brumo_2025",
            "MathArena/hmmt_feb_2025",
        ]:
            problem, answer = row["problem"], row["answer"]
        elif data_source == "dataset/Polaris-Dataset-Hard":
            problem, answer = row["problem"], row["answer"]

        prompt = problem + answer_format
        data = {
            "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025, polaris-dataset-hard
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "spo_tool_agent",
        }
        return data

    def map_fn2(self, row: dict):
        content = row["prompt"]
        row["prompt"] = [{"role": "user", "content": content + answer_format}]
        row["agent_name"] = "spo_tool_agent"
        return row


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    # Check format: if more than one "</think>" tag, score should be zero
    if solution_str.count("</think>") != 1:
        return {"score": 0, "acc": False, "pred": ""}

    # Check if there are <code> or <interpreter> blocks after </think>
    think_end_pos = solution_str.find("</think>")
    if think_end_pos != -1:
        after_think = solution_str[think_end_pos + len("</think>") :]
        if "<code>" in after_think or "<interpreter>" in after_think:
            return {"score": 0, "acc": False, "pred": ""}

    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # Modify to 0, +1 reward
    if result["score"] < 0:
        result["score"] = 0

    if result["pred"] is None:
        result["pred"] = ""

    return result
