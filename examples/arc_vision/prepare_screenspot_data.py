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
"""
Prepare ScreenSpot dataset for Arc Vision RL training.

This script converts the official ScreenSpot dataset from Hugging Face
(https://huggingface.co/datasets/rootsautomation/ScreenSpot) 
to VERL-compatible parquet format.
"""

import argparse
import json
import os
from typing import Dict, List, Any

import datasets
import pandas as pd
from PIL import Image
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs


def create_reasoning_prompt(instruction: str) -> str:
    """Create a reasoning-enhanced prompt for UI detection.
    
    This prompt encourages the model to think about whether it needs tools
    before attempting detection.
    """
    prompt = f"""{instruction}

First, analyze the image and describe what you observe about the target element:
<reasoning>
- Is the element clearly visible or partially obscured?
- Is it small, blurry, or low contrast?
- What challenges do you face in locating it?
- Do you need to use tools to see it better?
</reasoning>

Then provide the bounding box coordinates [x1, y1, x2, y2] in normalized format (0-1).
If you need to use tools, you can call:
- zoom_ui_element: To zoom into a region for better visibility
- wait_for_ui: To wait for elements to load
- inspect_element: To get additional information about UI structure"""
    
    return prompt


def process_screenspot_sample(sample: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    """Process a single ScreenSpot sample into VERL format.
    
    Args:
        sample: Raw ScreenSpot sample
        idx: Sample index
        split: Dataset split (train/validation/test)
    
    Returns:
        Processed sample in VERL format
    """
    # Extract fields from ScreenSpot
    instruction = sample.get("instruction", "")
    image = sample.get("image")
    bbox = sample.get("bbox", [0, 0, 0, 0])  # [x_min, y_min, x_max, y_max]
    
    # Ensure bbox is in normalized format (0-1)
    # ScreenSpot bboxes should already be normalized, but let's verify
    bbox_normalized = [
        max(0, min(1, float(bbox[0]))),  # x1
        max(0, min(1, float(bbox[1]))),  # y1
        max(0, min(1, float(bbox[2]))),  # x2
        max(0, min(1, float(bbox[3])))   # y2
    ]
    
    # Create reasoning-enhanced prompt
    enhanced_prompt = create_reasoning_prompt(instruction)
    
    # Format as chat messages
    messages = [
        {
            "role": "user",
            "content": enhanced_prompt
        }
    ]
    
    # Create VERL-compatible record
    record = {
        "data_source": "rootsautomation/ScreenSpot",
        "prompt": messages,
        "images": [image],  # PIL Image object
        "ability": "ui_detection",
        "reward_model": {
            "style": "arc_vision",
            "ground_truth": json.dumps(bbox_normalized),
            "confidence_threshold": 0.7,
            "reward_weights": {
                "task": 0.6,
                "tool": 0.3,
                "gate": 0.1
            }
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "original_instruction": instruction,
            "original_bbox": bbox,
            "element_type": sample.get("element_type", "unknown"),
            "screenshot_id": sample.get("screenshot_id", f"{split}_{idx}")
        }
    }
    
    return record


def main():
    parser = argparse.ArgumentParser(description="Prepare ScreenSpot dataset for Arc Vision RL")
    parser.add_argument("--local_dir", default="~/data/arc_vision/screenspot", 
                        help="Local directory to save processed data")
    parser.add_argument("--hdfs_dir", default=None, 
                        help="Optional HDFS directory to copy data to")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for debugging)")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        help="Dataset splits to process")
    
    args = parser.parse_args()
    
    # Expand local directory path
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    print("Loading ScreenSpot dataset from Hugging Face...")
    
    # Load the official ScreenSpot dataset
    dataset_dict = datasets.load_dataset("rootsautomation/ScreenSpot")
    
    for split in args.splits:
        if split not in dataset_dict:
            print(f"Warning: Split '{split}' not found in dataset. Skipping.")
            continue
        
        print(f"\nProcessing {split} split...")
        dataset = dataset_dict[split]
        
        # Limit samples if specified
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        
        print(f"Total samples in {split}: {len(dataset)}")
        
        # Process samples
        records = []
        for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split}")):
            try:
                record = process_screenspot_sample(sample, idx, split)
                records.append(record)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        print(f"Successfully processed {len(records)} samples")
        
        # Convert to DataFrame and save as parquet
        df = pd.DataFrame(records)
        output_file = os.path.join(local_dir, f"{split}.parquet")
        df.to_parquet(output_file)
        print(f"Saved {split} data to: {output_file}")
        
        # Print sample statistics
        print(f"\n{split} statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Average instruction length: {df['extra_info'].apply(lambda x: len(x['original_instruction'])).mean():.1f} chars")
        
        # Check bbox distribution
        bboxes = df['reward_model'].apply(lambda x: json.loads(x['ground_truth']))
        bbox_areas = bboxes.apply(lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        print(f"  Average bbox area: {bbox_areas.mean():.3f}")
        print(f"  Min bbox area: {bbox_areas.min():.3f}")
        print(f"  Max bbox area: {bbox_areas.max():.3f}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir:
        print(f"\nCopying data to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print("Data copied to HDFS successfully")
    
    print("\nData preparation complete!")
    print(f"Local data directory: {local_dir}")
    
    # Print example usage
    print("\nTo use this data in training, add to your config:")
    print(f"  data.train_files: {os.path.join(local_dir, 'train.parquet')}")
    print(f"  data.val_files: {os.path.join(local_dir, 'validation.parquet')}")
    print(f"  data.test_files: {os.path.join(local_dir, 'test.parquet')}")


if __name__ == "__main__":
    main()