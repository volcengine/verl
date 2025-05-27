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
Preprocess the UGround dataset to parquet format
"""

import argparse
import io
import os
import json
import logging

import pandas as pd
from PIL import Image
from datasets import Dataset, Sequence
from datasets import Image as ImageData

from verl.utils.hdfs_io import copy, makedirs
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import smart_resize


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)


def extract_bounding_box(original_action):
    """
    Extract bounding box coordinates from original_action string.

    Args:
        original_action (str): String containing bounding box information

    Returns:
        tuple: (x1, y1, x2, y2) coordinates or None if not found
    """
    try:
        # Split by first ']' and get the JSON part
        json_str = original_action.split("]", 1)[1]
        # Parse JSON
        data = json.loads(json_str)
        # Extract box_model coordinates
        box = data.get("box_model")
        if box and len(box) == 4:
            return tuple(float(x) for x in box)
    except (IndexError, json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def read_parquet_file(file_path):
    """
    Read a parquet file containing viewport images and action data.

    Args:
        file_path (str): Path to the parquet file

    Returns:
        pd.DataFrame: DataFrame containing the data with columns:
            - viewport: Image bytes
            - vision_compatible_action: String describing the action
            - action_desc: String describing the action in more detail
            - original_action: String containing bounding box information
    """
    df = pd.read_parquet(file_path)

    # Verify required columns exist
    required_columns = [
        "viewport",
        "vision_compatible_action",
        "action_desc",
        "original_action",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def get_xy_resize_ratio(image):
    """
    Get the resize ratio of the image.
    """
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=PROCESSOR.image_processor.patch_size
        * PROCESSOR.image_processor.merge_size,
        min_pixels=PROCESSOR.image_processor.min_pixels,
        max_pixels=PROCESSOR.image_processor.max_pixels,
    )

    return resized_height / image.height, resized_width / image.width


def process_data(df, split):
    """
    Process the data into the required format.

    Args:
        df (pd.DataFrame): Input DataFrame
        split (str): Dataset split name ('train' or 'test')

    Returns:
        pd.DataFrame: Processed DataFrame
    """

    def process_fn(row, idx):
        # Extract bounding box from original_action
        image = Image.open(io.BytesIO(row["viewport"]))
        x_resize_ratio, y_resize_ratio = get_xy_resize_ratio(image)
        bounding_box = extract_bounding_box(row["original_action"])
        if bounding_box is not None:
            # Convert the bounding box based on the resized image size
            bounding_box = [
                bounding_box[0] * x_resize_ratio,
                bounding_box[1] * y_resize_ratio,
                bounding_box[2] * x_resize_ratio,
                bounding_box[3] * y_resize_ratio,
            ]
        ground_truth = {
            "action": row["vision_compatible_action"],
            "bbox": bounding_box,
        }

        action_desc = row["action_desc"].strip()
        if action_desc.endswith("assistant"):
            action_desc = action_desc[:-9].strip()

        data = {
            "data_source": "uground",
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Map the user action description to the action with coordinates in the UI image. "
                        "Think step by step before you answer. The reasoning process MUST BE enclosed within <think> </think> tags. "
                        "The action MUST BE put in <answer> </answer> tags. "
                        "The candidate actions include the following:\n"
                        "- mouse_click(x, y): Click on the coordinate (x, y)\n"
                        "- mouse_move(x, y): Move the mouse cursor to the coordinate (x, y)\n"
                        "- scroll(x, y): Scroll by the distance in the x and/or y axis\n"
                        "- keyboard_type('content'): Type the content in the quotes\n"
                        "- select_option(x, y): Select the option at coordinate (x, y)\n"
                        "<image> Action description: " + action_desc
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
                "answer": row["vision_compatible_action"],
                "question": action_desc,
                "bounding_box": bounding_box,
            },
        }
        return data

    for idx, row in df.iterrows():
        data = process_fn(row, idx)
        if len(data["prompt"][0]["content"]) > 2000:
            logging.warning(
                "Too long prompt: " + str(len(data["prompt"][0]["content"]))
            )
            # Filter out the data with too long prompt
            continue
        yield data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", required=True, help="Path to the input parquet file"
    )
    parser.add_argument("--local_dir", default="~/data/uground")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "dev"],
        help="Dataset split",
    )

    args = parser.parse_args()

    # Save to local directory
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    output_file = os.path.join(local_dir, f"{args.split}.parquet")

    # Read the input parquet file and save it to dataset.
    df = read_parquet_file(args.input_file)
    # Have to do this for the PIL Image object, otherwise causing conversion type error.
    dataset = Dataset.from_generator(
        process_data, gen_kwargs={"df": df, "split": args.split}
    )
    dataset = dataset.cast_column("images", Sequence(ImageData()))
    dataset.to_parquet(output_file)

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
