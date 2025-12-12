#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert math problem dataset to veRL's RL training format

Input format example:
{
    'problem': 'Math problem description',
    'answer': 'Answer',
    'solution': 'Detailed solution process'
}

Output format:
{
    "data_source": "custom_math_dataset",
    "prompt": [{"role": "user", "content": "problem + output format instruction"}],
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": "answer"}
}
"""

import json
import argparse
import os
import re
from typing import List, Dict, Any
import pandas as pd
import datasets
from pathlib import Path


def extract_final_answer(answer_str: str) -> str:
    """
    Extract final answer from answer string
    Handle various formats: fractions, decimals, integers, etc.
    
    Args:
        answer_str: Original answer string, e.g., "-\\frac{2}{3}" or "26"
    
    Returns:
        Processed answer string
    """
    # Remove whitespace
    answer_str = answer_str.strip()
    
    # If it's a simple number, return directly
    if re.match(r'^-?\d+\.?\d*$', answer_str):
        return answer_str
    
    # Handle fraction format \frac{a}{b}
    frac_pattern = r'(-?)\\frac\{(.+?)\}\{(.+?)\}'
    frac_match = re.match(frac_pattern, answer_str)
    if frac_match:
        sign, numerator, denominator = frac_match.groups()
        # Keep fraction form as some math problems need exact fraction answers
        return f"{sign}{numerator}/{denominator}"
    
    # Handle other LaTeX formats (add more if needed)
    # You can add more processing logic based on actual data
    
    return answer_str


def convert_to_rl_format(item: Dict[str, Any], data_source: str = "custom_math_dataset") -> Dict[str, Any]:
    """
    Convert a single math problem to RL training format
    
    Args:
        item: Dict containing problem, answer, solution
        data_source: Data source identifier
    
    Returns:
        Dict in veRL RL dataset format
    """
    # Extract problem and answer
    problem = item['problem']
    answer = extract_final_answer(item['answer'])
    
    # Add output format instruction
    # This instruction is important - it tells the model how to format output for easy answer extraction
    instruction_suffix = '\n\nPlease think step by step and provide the final answer with "The answer is:" prefix.'
    
    # Build prompt
    prompt_content = problem + instruction_suffix
    
    # Build final format
    rl_format = {
        "data_source": data_source,
        "prompt": [
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer  # ground_truth must be a list
        },
        "extra_info": {
            "original_problem": problem,
            "original_answer": item['answer'],
            "solution": item.get('solution', '')
        }
    }
    
    return rl_format


def process_json_file(input_file: str, output_dir: str, data_source: str = "custom_math_dataset"):
    """
    Process JSON file and convert to Parquet format
    
    Args:
        input_file: Input JSON file path
        output_dir: Output directory
        data_source: Data source identifier
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading file: {input_file}")
    
    # Read JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # If data is a single dict, convert to list
    if isinstance(data, dict):
        data = [data]
    
    print(f"Found {len(data)} items")
    
    # Convert data format
    converted_data = []
    for idx, item in enumerate(data):
        try:
            converted_item = convert_to_rl_format(item, data_source)
            converted_item['extra_info']['index'] = idx
            converted_data.append(converted_item)
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            print(f"Problem data: {item}")
            continue
    
    print(f"Successfully converted {len(converted_data)} items")
    
    # Convert to DataFrame
    df = pd.DataFrame(converted_data)
    
    # Create HuggingFace Dataset
    dataset = datasets.Dataset.from_pandas(df)
    
    # Save as Parquet file
    output_file = os.path.join(output_dir, "train.parquet")
    dataset.to_parquet(output_file)
    
    print(f"Data saved to: {output_file}")
    
    # Optional: Also save a JSON format for inspection
    json_output = os.path.join(output_dir, "train_sample.json")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(converted_data[:5], f, ensure_ascii=False, indent=2)
    print(f"First 5 samples saved to: {json_output} (for inspection)")
    
    return output_file


def split_dataset(input_file: str, output_dir: str, train_ratio: float = 0.9):
    """
    Split dataset into train and validation sets
    
    Args:
        input_file: Input JSON file path
        output_dir: Output directory
        train_ratio: Training set ratio
    """
    import random
    
    # Read data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    # Shuffle randomly
    random.shuffle(data)
    
    # Split data
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"💡 Train set: {len(train_data)} items")
    print(f"💡 Validation set: {len(val_data)} items")
    
    # Process train and validation sets separately
    train_converted = []
    val_converted = []
    
    for idx, item in enumerate(train_data):
        converted = convert_to_rl_format(item)
        converted['extra_info']['split'] = 'train'
        converted['extra_info']['index'] = idx
        train_converted.append(converted)
    
    for idx, item in enumerate(val_data):
        converted = convert_to_rl_format(item)
        converted['extra_info']['split'] = 'val'
        converted['extra_info']['index'] = idx
        val_converted.append(converted)
    
    # Save as Parquet
    train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(train_converted))
    val_dataset = datasets.Dataset.from_pandas(pd.DataFrame(val_converted))
    
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "val.parquet"))
    
    print(f"✅ Train set saved to: {os.path.join(output_dir, 'train.parquet')}")
    print(f"✅ Validation set saved to: {os.path.join(output_dir, 'val.parquet')}")


def main():
    parser = argparse.ArgumentParser(description="Convert math problem dataset to veRL RL training format")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Input JSON file path")
    parser.add_argument("--output", "-o", type=str, default="./output",
                        help="Output directory path (default: ./output)")
    parser.add_argument("--data-source", type=str, default="DeepScaleR",
                        help="Data source identifier (default: DeepScaleR)")
    parser.add_argument("--split", action="store_true",
                        help="Whether to split into train and validation sets")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Training set ratio (default: 0.9)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    # Execute conversion
    if args.split:
        split_dataset(args.input, args.output, args.train_ratio)
    else:
        process_json_file(args.input, args.output, args.data_source)


if __name__ == "__main__":
    main()