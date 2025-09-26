# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


# huggingface-cli download nvidia/OpenMathReasoning --repo-type dataset --include data/genselect* --local-dir /path/to/nvidia/OpenMathReasoning

import argparse
import datasets
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/open_math_reasoning", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "nvidia/OpenMathReasoning"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            solution = example.pop("generated_solution")

            extra_info = {}
            for key, value in example.items():
                extra_info[key] = value
            example.clear()

            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                        "loss_mask": 0
                    },
                    {
                        "role": "assistant",
                        "content": solution,
                        "loss_mask": 1
                    },
                ],
                "extra_info": extra_info
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("genselect"), with_indices=True)
    genselect_dataset = dataset['genselect']
    local_save_dir = args.local_save_dir
    genselect_dataset.to_parquet(os.path.join(local_save_dir, "genselect_dataset.parquet"))
