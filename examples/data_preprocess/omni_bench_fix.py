# Copyright 2025 Individual Contributor: TomQunChaoA
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
Preprocess the Omni_Bench_fix dataset to parquet format
"""

import argparse
import os

import datasets
import librosa
from qwen_omni_utils.v2_5.audio_process import SAMPLE_RATE

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/lmms-lab/Omni_Bench_fix")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "lmms-lab/Omni_Bench_fix"
    dataset = datasets.load_dataset(
        data_source,
        split="train",
    )  # this dataset has no test split
    train_dataset, test_dataset = dataset.train_test_split(test_size=0.2).values()
    instruction_following = ""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = "<image> <audio> " + example.pop("question")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            image = example.pop("image")
            audio = example.pop("audio")
            y16k_audio = librosa.resample(audio["array"], orig_sr=audio["sampling_rate"], target_sr=SAMPLE_RATE)

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are Qwen, a virtual human developed by the Qwen Team, "
                                "Alibaba Group, capable of perceiving auditory and visual inputs, "
                                "as well as generating text and speech.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": [image],
                "audios": [y16k_audio],
                "ability": "multimodal",
                "reward_model": {"style": "rule", "ground_truth": {"target": answer}},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset: datasets.IterableDataset = train_dataset
    to_remove_columns = list(
        set(train_dataset.column_names)
        - set(
            [
                "audios",
                "images",
                "ability",
                "extra_info",
                "data_source",
                "prompt",
                "ability",
                "reward_model",
                "question",
                "answer",
                "image",
                "audio",
            ]
        )
    )
    train_dataset_new = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    train_dataset_new = train_dataset_new.remove_columns(to_remove_columns)
    test_dataset_new = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)
    test_dataset_new = test_dataset_new.remove_columns(to_remove_columns)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_dataset_new.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset_new.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
