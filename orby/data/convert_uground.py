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
import math

import datasets
from datasets import Sequence
from datasets import Image as ImageData
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import smart_resize
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs
from verl.utils import hf_processor

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
# PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH)
PROCESSOR = hf_processor(MODEL_PATH, use_fast=True)

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


def save_in_chunks(all_data, output_dir, prefix, max_examples_per_file=12500):
    """Save processed data in multiple parquet files"""
    os.makedirs(output_dir, exist_ok=True)
    
    file_counter = 0
    total_examples = 0
    
    # Combine all datasets first to get total count
    combined_data = datasets.concatenate_datasets(all_data)
    total_examples = len(combined_data)
    
    print(f"Saving {total_examples} examples in chunks of {max_examples_per_file}...", flush=True)
    
    # Save in chunks
    for start_idx in range(0, total_examples, max_examples_per_file):
        end_idx = min(start_idx + max_examples_per_file, total_examples)
        chunk = combined_data.select(range(start_idx, end_idx))
        
        output_file = os.path.join(output_dir, f"{prefix}_part_{file_counter:04d}.parquet")
        chunk.to_parquet(output_file)
        print(f"Saved {len(chunk)} examples to {output_file}", flush=True)
        
        file_counter += 1
    
    return file_counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/uground")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_files", default="shard_*.parquet")
    parser.add_argument("--output_filename", default="train")
    parser.add_argument("--prompt_format", choices=["thinking", "sft"], required=True, default="thinking")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of examples to process at once")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process (for testing)")
    parser.add_argument("--max_examples_per_file", type=int, default=12500, help="Maximum examples per output parquet file")

    args = parser.parse_args()

    data_source = "osunlp/UGround-V1-Data-Box"
    print(f"Loading the {data_source} dataset from huggingface in streaming mode...", flush=True)
    
    # Load in streaming mode
    dataset = datasets.load_dataset(data_source, data_files=args.data_files, streaming=True)
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

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            data = {
                "data_source": "uground",
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

                data["response"] = [
                    {
                        "role": "assistant",
                        "content": f"<answer>{center_x:.0f} {center_y:.0f}</answer>"
                    }
                ]
            return data

        return process_fn
    
    def process_in_chunks(streaming_dataset, chunk_size=1000):
        """Process streaming dataset in chunks to manage memory"""
        chunk = []
        total_processed = 0
        
        for i, example in enumerate(streaming_dataset):
            if args.max_examples and total_processed >= args.max_examples:
                break
                
            chunk.append(example)
            
            if len(chunk) >= chunk_size:
                print(f"Processing chunk {total_processed//chunk_size + 1}, examples {total_processed}-{total_processed + len(chunk)}", flush=True)
                
                # Convert chunk to Dataset for processing
                chunk_dataset = Dataset.from_list(chunk)
                
                # Process the chunk
                processed_chunk = chunk_dataset.map(
                    function=make_map_fn("train"), 
                    with_indices=True, 
                    num_proc=4  # Reduced from 16 to manage memory
                )
                processed_chunk = processed_chunk.cast_column("images", Sequence(ImageData()))
                
                yield processed_chunk, total_processed
                
                total_processed += len(chunk)
                chunk = []
        
        # Process remaining examples
        if chunk:
            print(f"Processing final chunk, examples {total_processed}-{total_processed + len(chunk)}", flush=True)
            chunk_dataset = Dataset.from_list(chunk)
            processed_chunk = chunk_dataset.map(
                function=make_map_fn("train"), 
                with_indices=True, 
                num_proc=4
            )
            processed_chunk = processed_chunk.cast_column("images", Sequence(ImageData()))
            yield processed_chunk, total_processed

    local_dir = os.path.expanduser(args.local_dir)
    if args.prompt_format == "sft":
        local_dir += "_sft"
    
    print(f"Saving to {local_dir}...", flush=True)
    os.makedirs(local_dir, exist_ok=True)

    if args.prompt_format == "sft":
        # For SFT, we need to handle train/test split differently with streaming
        all_train_data = []
        all_test_data = []
        
        for chunk_dataset, chunk_start in process_in_chunks(dataset, args.chunk_size):
            # Split each chunk into train/test
            chunk_split = chunk_dataset.train_test_split(train_size=0.8, seed=42)
            all_train_data.append(chunk_split['train'])
            all_test_data.append(chunk_split['test'])
        
        # Save train data in multiple files
        if all_train_data:
            train_dir = os.path.join(local_dir, "train")
            train_files = save_in_chunks(all_train_data, train_dir, "train", args.max_examples_per_file)
            print(f"Saved train data in {train_files} files", flush=True)
            
            test_dir = os.path.join(local_dir, "test")
            test_files = save_in_chunks(all_test_data, test_dir, "test", args.max_examples_per_file // 4)  # Smaller test files
            print(f"Saved test data in {test_files} files", flush=True)
    else:
        dataset = dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=16)
        dataset = dataset.cast_column("images", Sequence(ImageData()))

        local_dir = os.path.expanduser(args.local_dir)
        os.makedirs(local_dir, exist_ok=True)

        dataset.to_parquet(os.path.join(local_dir, f"{args.output_filename}.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
