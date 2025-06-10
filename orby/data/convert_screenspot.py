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
Preprocess the Screenspot dataset to parquet format
"""

import argparse
import io
import os

import datasets
from datasets import Sequence
from datasets import Image as ImageData
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import smart_resize

from verl.utils.hdfs_io import copy, makedirs
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from orby.utils.dataset.qwen_agent_function_call import ComputerUse


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)

_SOURCE_MAP = {
    "ios": "mobile",
    "android": "mobile",
    "windows": "desktop",
    "macos": "desktop",
}


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
    parser.add_argument("--local_dir", default="~/data/screenspot")
    parser.add_argument("--hdfs_dir", default=None)
    # Check below for the thinking format and qwen format.
    # Thining format is a simple prompt that asks the model to think step by step
    # and then answer the question.
    # Qwen format implementation was referenced from
    # https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/computer_use.ipynb
    parser.add_argument(
        "--prompt_format",
        choices=["thinking", "qwen", "sft"],
        default="thinking",
        help="Select prompt format: ['thinking', 'qwen', 'sft']",
    )
    
    args = parser.parse_args()

    print(f"Prompt format: {args.prompt_format}", flush=True)

    data_source = "rootsautomation/ScreenSpot"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source)

    test_dataset = dataset["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            image = example.pop("image")
            instruction = example.pop("instruction").strip()
            bbox = example.pop("bbox")
            data_type = example.pop("data_type")
            data_source = example.pop("data_source")
            device = _SOURCE_MAP.get(data_source, "web")

            # Get image and resize ratios
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            resized_height, resized_width = get_resized_wh(image)

            # Adjust bbox based on resize ratios
            bbox = [
                bbox[0] * resized_width,
                bbox[1] * resized_height,
                bbox[2] * resized_width,
                bbox[3] * resized_height,
            ]

            ground_truth = {
                "bbox": bbox,
                "data_type": data_type,
                "device": device,
            }

            data = {
                "data_source": "screenspot",
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
                "images": [image],
            }

            # Create prompt based on selected format
            if args.prompt_format == "thinking":
                data["prompt"] = [
                    {
                        "role": "user",
                        "content": (
                            "Map the user instruction to the coordinates in the UI image. "
                            "Think step by step before you answer. The reasoning process MUST BE enclosed within <think> </think> tags. "
                            "The coordinate x and y MUST BE put in <answer> </answer> tags, separeted by space. "
                            "<image> Instruction: " + instruction
                        ),
                    },
                ]
            elif args.prompt_format == "sft":
                data["prompt"] = [
                    {
                        "role": "user",
                        "content": (
                            "<image> Instruction: " + instruction
                        ),
                    },
                ]

            else:  # qwen format
                prompt = NousFnCallPrompt().preprocess_fncall_messages(
                    messages=[
                        Message(
                            role="system",
                            content=[ContentItem(text="You are a helpful assistant.")],
                        ),
                        Message(
                            role="user",
                            content=[
                                ContentItem(text=instruction + "<image>"),
                            ],
                        ),
                    ],
                    functions=[
                        ComputerUse(
                            cfg={
                                "display_width_px": resized_width,
                                "display_height_px": resized_height,
                            }
                        ).function
                    ],
                    lang=None,
                )

                prompt = [msg.model_dump() for msg in prompt]
                for message in prompt:
                    # Replace the list of content to a string.
                    content = "".join(m["text"] for m in message["content"])
                    message["content"] = content

                data["prompt"] = prompt
                data["reward_model"]["format"] = "qwen"

            data.update(example)
            return data

        return process_fn

    test_dataset = test_dataset.map(
        function=make_map_fn("test"), with_indices=True, num_proc=16
    )
    test_dataset = test_dataset.cast_column("images", Sequence(ImageData()))

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
