# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/download.py


import argparse

from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Download files from a Hugging Face dataset repository.")
parser.add_argument("--repo_id", type=str, default="PeterJinGo/wiki-18-e5-index", help="Hugging Face repository ID")
parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")

args = parser.parse_args()

repo_id = "PeterJinGo/wiki-18-e5-index"
for file in ["part_aa", "part_ab"]:
    hf_hub_download(
        repo_id=repo_id,
        filename=file,  # e.g., "e5_Flat.index"
        repo_type="dataset",
        local_dir=args.save_path,
    )

repo_id = "PeterJinGo/wiki-18-corpus"
hf_hub_download(
    repo_id=repo_id,
    filename="wiki-18.jsonl.gz",
    repo_type="dataset",
    local_dir=args.save_path,
)
