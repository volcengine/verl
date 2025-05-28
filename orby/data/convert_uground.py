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
Preprocess the Uground dataset to parquet format
"""

import argparse
import io
import json
import os

import datasets
from datasets import Sequence
from datasets import Image as ImageData
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import smart_resize

from verl.utils.hdfs_io import copy, makedirs


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)


def get_resized_wh(image):
    """
    Get the resized width and height of the image.
    """
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=PROCESSOR.image_processor.patch_size
        * PROCESSOR.image_processor.merge_size,
        min_pixels=PROCESSOR.image_processor.min_pixels,
        max_pixels=PROCESSOR.image_processor.max_pixels,
    )

    return resized_height, resized_width


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/uground")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_files", default="shard_0000.parquet")
    parser.add_argument("--output_filename", default="train")

    args = parser.parse_args()

    data_source = "osunlp/UGround-V1-Data-Box"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, data_files=args.data_files)

    dataset = dataset["train"]

    def make_map_fn(split):
        def process_fn(example, idx):
            image = example.pop("image")
            conversation = example.pop("conversations").strip()
            # Use the first message for now. Uground has multiple grounding
            # commands / groundtruths in the conversation.
            command, label = json.loads(conversation)[:2]
            assert command["from"] == "human" and label["from"] == "gpt"
            instruction = command["value"]
            label_text = label["value"]

            # Parse the label text as "(x1, y1, x2, y2)" format
            label_text = label_text.strip("()")
            bbox = [int(x.strip()) for x in label_text.split(",")]
            assert len(bbox) == 4, f"Expected 4 coordinates, got {len(bbox)}"

            # Get image and resize ratios
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            resized_height, resized_width = get_resized_wh(image)

            # Adjust bbox based on resize ratios. Uground labels range from
            # [0, 999]
            bbox = [
                bbox[0] * resized_width / 999.0,
                bbox[1] * resized_height / 999.0,
                bbox[2] * resized_width / 999.0,
                bbox[3] * resized_height / 999.0,
            ]

            ground_truth = {
                "bbox": bbox,
            }

            data = {
                "data_source": "uground",
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            "Map the user instruction to the coordinates in the UI image. "
                            "Think step by step before you answer. The reasoning process MUST BE enclosed within <think> </think> tags. "
                            "The coordinate x and y MUST BE put in <answer> </answer> tags, separeted by space. "
                            "<image> Instruction: " + instruction
                        ),
                    },
                ],
                "images": [image],
                "ability": "vision",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": instruction,
                    "bounding_box": bbox,
                },
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=16)
    dataset = dataset.cast_column("images", Sequence(ImageData()))

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    dataset.to_parquet(os.path.join(local_dir, f"{args.output_filename}.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
