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


def process_screenspot_sample(sample: Dict[str, Any], idx: int, split: str, image_dir: str) -> Dict[str, Any]:
    """Process a single ScreenSpot sample into VERL format.
    
    Args:
        sample: Raw ScreenSpot sample
        idx: Sample index
        split: Dataset split (train/validation/test)
        image_dir: Directory to save images
    
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
    
    # Save image to disk
    image_filename = f"{split}_{idx}.png"
    image_path = os.path.join(image_dir, image_filename)
    if image:
        image.save(image_path)
    
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
        "prompt": json.dumps(messages),  # Serialize to JSON string
        "images": [image_path],  # Store path instead of PIL object
        "ability": "ui_detection",
        "reward_model": json.dumps({  # Serialize entire dict
            "style": "arc_vision",
            "ground_truth": bbox_normalized,
            "confidence_threshold": 0.7,
            "reward_weights": {
                "task": 0.6,
                "tool": 0.3,
                "gate": 0.1
            }
        }),
        "extra_info": json.dumps({  # Serialize entire dict
            "split": split,
            "index": idx,
            "original_instruction": instruction,
            "original_bbox": bbox,
            "element_type": sample.get("element_type", "unknown"),
            "screenshot_id": sample.get("screenshot_id", f"{split}_{idx}")
        })
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
    parser.add_argument("--split_test_data", action="store_true",
                        help="Split test data into train/validation/test (60/20/20)")
    
    args = parser.parse_args()
    
    # Expand local directory path
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    print("Loading ScreenSpot dataset from Hugging Face...")
    
    # Load the official ScreenSpot dataset
    dataset_dict = datasets.load_dataset("rootsautomation/ScreenSpot")
    
    # Check what splits are actually available
    available_splits = list(dataset_dict.keys())
    print(f"Available splits in ScreenSpot: {available_splits}")
    
    # If only test split exists and user wants to create train/val splits
    if len(available_splits) == 1 and 'test' in available_splits and args.split_test_data:
        print("\nOnly test split found. Creating train/validation/test splits from test data...")
        test_dataset = dataset_dict['test']
        
        # Limit samples if specified
        if args.max_samples:
            test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
        
        total_samples = len(test_dataset)
        print(f"Total samples to split: {total_samples}")
        
        # Calculate split sizes (60/20/20)
        train_size = int(0.6 * total_samples)
        val_size = int(0.2 * total_samples)
        test_size = total_samples - train_size - val_size
        
        print(f"Split sizes - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
        
        # Create splits
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_samples))
        
        splits_to_process = [
            ('train', test_dataset.select(train_indices)),
            ('validation', test_dataset.select(val_indices)),
            ('test', test_dataset.select(test_indices))
        ]
    else:
        # Process original splits
        splits_to_process = []
        for split in args.splits:
            if split not in dataset_dict:
                print(f"Warning: Split '{split}' not found in dataset. Skipping.")
                continue
            
            dataset = dataset_dict[split]
            
            # Limit samples if specified
            if args.max_samples:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            
            splits_to_process.append((split, dataset))
    
    # Process each split
    for split, dataset in splits_to_process:
        print(f"\nProcessing {split} split...")
        print(f"Total samples in {split}: {len(dataset)}")
        
        # Create directory for images
        image_dir = os.path.join(local_dir, f"{split}_images")
        os.makedirs(image_dir, exist_ok=True)
        
        # Process samples
        records = []
        for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split}")):
            try:
                record = process_screenspot_sample(sample, idx, split, image_dir)
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
        print(f"  Images saved to: {image_dir}")
        
        # Parse extra_info to get instruction length
        extra_infos = df['extra_info'].apply(json.loads)
        print(f"  Average instruction length: {extra_infos.apply(lambda x: len(x['original_instruction'])).mean():.1f} chars")
        
        # Check bbox distribution
        reward_models = df['reward_model'].apply(json.loads)
        bboxes = reward_models.apply(lambda x: x['ground_truth'])
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