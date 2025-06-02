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
Preprocess the Screenspot Pro dataset to parquet format
"""

import argparse
import io
import json
import os
import logging
import glob

import datasets
from datasets import Sequence
from datasets import Image as ImageData
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import smart_resize
import pyarrow.parquet as pq

from verl.utils.hdfs_io import copy, makedirs


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)

_PLATFORM_MAP = {
    "ios": "mobile",
    "android": "mobile",
    "windows": "desktop",
    "macos": "desktop",
    "linux": "desktop",
    "web": "web",
}


def get_resized_ratio(image):
    """
    Get the resize ratios for width and height of the image.

    Returns:
        Tuple of (height_ratio, width_ratio) where each ratio is the resized dimension
        divided by the original dimension.
    """
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=PROCESSOR.image_processor.patch_size
        * PROCESSOR.image_processor.merge_size,
        min_pixels=PROCESSOR.image_processor.min_pixels,
        max_pixels=PROCESSOR.image_processor.max_pixels,
    )

    height_ratio = resized_height / image.height
    width_ratio = resized_width / image.width

    return height_ratio, width_ratio


def process_json_file(json_path, image_dir, split):
    """
    Process a single JSON file and return a list of processed examples.

    Args:
        json_path: Path to the JSON file
        image_dir: Directory containing the images
        split: Dataset split name (e.g., "train", "test")

    Returns:
        List of processed examples
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    processed_examples = []
    for idx, example in enumerate(data):
        # Load image from file
        img_path = os.path.join(image_dir, example["img_filename"])
        try:
            image = Image.open(img_path)
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or "PNG")
            img_byte_arr = img_byte_arr.getvalue()
        except Exception as e:
            logging.warning(f"Failed to load image {img_path}: {e}")
            continue

        # Get image resize ratios
        height_ratio, width_ratio = get_resized_ratio(image)

        # Adjust bbox based on resize ratios
        bbox = example["bbox"]
        bbox = [
            bbox[0] * width_ratio,
            bbox[1] * height_ratio,
            bbox[2] * width_ratio,
            bbox[3] * height_ratio,
        ]

        device = _PLATFORM_MAP.get(example["platform"], "web")

        ground_truth = {
            "bbox": bbox,
            "data_type": example["ui_type"],
            "device": device,
            "application": example["application"],
        }

        data = {
            "data_source": "screenspot_pro",
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        "Map the user instruction to the coordinates in the UI image. "
                        "Think step by step before you answer. The reasoning process MUST BE enclosed within <think> </think> tags. "
                        "The coordinate x and y MUST BE put in <answer> </answer> tags, separeted by space. "
                        "<image> Instruction: " + example["instruction"]
                    ),
                },
            ],
            "images": [img_byte_arr],
            "ability": "vision",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "question": example["instruction"],
                "bounding_box": bbox,
                "application": example["application"],
                "platform": example["platform"],
                "ui_type": example["ui_type"],
            },
        }
        processed_examples.append(data)

    return processed_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/screenspot_pro")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--image_dir",
        default="~/data/screenspot_pro/images",
        help="Directory containing the images",
    )
    parser.add_argument(
        "--annotations_dir",
        default="~/data/screenspot_pro/annotations",
        help="Directory containing the annotation JSON files",
    )

    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    image_dir = os.path.expanduser(args.image_dir)
    annotations_dir = os.path.expanduser(args.annotations_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Find all JSON files in the annotations directory
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {annotations_dir}")

    all_examples = []
    shard = 0
    for json_file in json_files:
        logging.info(f"Processing {json_file}")
        examples = process_json_file(json_file, image_dir, "test")
        all_examples.extend(examples)
        if len(all_examples) > 750:
            # Convert to dataset
            dataset = datasets.Dataset.from_list(all_examples)
            dataset = dataset.cast_column("images", Sequence(ImageData()))

            # Save to parquet
            dataset.to_parquet(os.path.join(local_dir, f"test-{shard}.parquet"))
            shard += 1
            all_examples = []

    # Convert the final shard.
    if len(all_examples):
        # Convert to dataset
        dataset = datasets.Dataset.from_list(all_examples)
        dataset = dataset.cast_column("images", Sequence(ImageData()))

        # Save to parquet
        dataset.to_parquet(os.path.join(local_dir, f"test-{shard}.parquet"))

    files = [os.path.join(local_dir, f"test-{shard}.parquet") for shard in range(shard)]

    schema = pq.ParquetFile(files[0]).schema_arrow
    with pq.ParquetWriter(
        os.path.join(local_dir, "test.parquet"), schema=schema
    ) as writer:
        for file in files:
            writer.write_table(pq.read_table(file, schema=schema))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
