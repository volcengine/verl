# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
Reward calculation for CosyVoice2-0.5B.
"""

from __future__ import annotations

import os, re, warnings, json, time, argparse
from typing import List

import numpy as np
import requests


REWARD_SERVER_URL = "http://localhost:8000/v2/models/token2wav_asr/infer"


def _parse_ids(token_str: str) -> List[int]:
    return [int(t) for t in re.findall(r"<\|s_(\d+)\|>", token_str)]

def _remote_reward(tokens: List[int], ground_truth: str, timeout: float = 200.0) -> float:
    """Send token IDs and ground-truth text to the Triton server and get reward."""

    tokens_arr = np.array(tokens, dtype=np.int32).reshape(1, -1)
    lens_arr = np.array([[tokens_arr.shape[1]]], dtype=np.int32)

    gt_arr = np.array([ground_truth.encode("utf-8")], dtype=object)

    payload = {
        "inputs": [
            {
                "name": "TOKENS",
                "shape": list(tokens_arr.shape),
                "datatype": "INT32",
                "data": tokens_arr.tolist(),
            },
            {
                "name": "TOKEN_LENS",
                "shape": list(lens_arr.shape),
                "datatype": "INT32",
                "data": lens_arr.tolist(),
            },
            {
                "name": "GT_TEXT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [ground_truth],
            },
        ]
    }
    rsp = requests.post(
        REWARD_SERVER_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
        verify=False,
        params={"request_id": "0"},
    )
    rsp.raise_for_status()
    result = rsp.json()

    try:
        # Reward is returned as the first output
        return float(result["outputs"][0]["data"][0])
    except (KeyError, IndexError, TypeError):
        return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    *,
    debug_dump: bool = False,
) -> float:
    """Return reward in [0, 1] using the Triton ASR service.

    The reward is based on the pinyin-level WER between the ASR transcript
    produced from *solution_str* and the provided *ground_truth* text.
    """

    # Decode token IDs
    ids = _parse_ids(solution_str)

    # Query remote server for reward
    try:
        reward = _remote_reward(ids, ground_truth)
    except Exception as e:
        warnings.warn(f"Remote reward server error: {e}; returning 0.0")
        reward = 0.0

    if debug_dump:
        print(
            f"\033[92m[{data_source}] Remote reward: {reward:.4f}\033[0m"
        )

    return reward

# CLI quick test
if __name__ == "__main__":
    import sys
    
    def get_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Test TTS CER scoring with data from JSONL file",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            "--input", "-i",
            type=str,
            default="data/emilia_zh-cosy-tiny-test.jsonl",
            help="Path to input JSONL file"
        )
        
        parser.add_argument(
            "--max-samples", "-n",
            type=int,
            default=None,
            help="Maximum number of samples to process (default: all)"
        )
        
        parser.add_argument(
            "--no-interactive",
            action="store_true",
            help="Run in non-interactive mode (process all samples without prompts)"
        )
        
        
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode"
        )
        
        return parser.parse_args()
    
    def load_jsonl(file_path: str):
        """Load data from jsonl file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def code_to_solution_str(code_list: List[int]) -> str:
        """Convert code list to solution string format."""
        return ''.join([f"<|s_{code}|>" for code in code_list])
    
    # Parse command line arguments
    args = get_args()
    
    try:
        # Load data from jsonl file
        print(f"Loading data from: {args.input}")
        data_list = load_jsonl(args.input)
        print(f"Loaded {len(data_list)} samples")
        
        # Limit samples if specified
        if args.max_samples is not None:
            data_list = data_list[:args.max_samples]
            print(f"Processing first {len(data_list)} samples (limited by --max-samples)")
        
        # Process each sample
        begin_time = time.time()
        for i, sample in enumerate(data_list):
            print(f"\n--- Sample {i+1}/{len(data_list)} ---")
            print(f"Index: {sample.get('index', 'unknown')}")
            print(f"Text: {sample['text']}")
            
            # Extract required fields
            code_list = sample['code']
            ground_truth = sample['text']
            data_source = sample.get('index', f'sample_{i}')  # Use index as data_source
            
            # Convert code list to solution string
            solution_str = code_to_solution_str(code_list)
            print(f"Solution tokens: {len(code_list)} tokens")
            if args.debug:
                print(f"Solution string: {solution_str}")
            else:
                print(f"Solution string preview: {solution_str[:100]}..." if len(solution_str) > 100 else f"Solution string: {solution_str}")
            
            # Call compute_score function
            try:
                score = compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=None,
                    debug_dump=args.debug
                )
                print(f"Final Score: {score:.4f}")
            except Exception as e:
                print(f"Error computing score: {e}")
            
            # Ask user if they want to continue (for interactive mode)
            if not args.no_interactive and i < len(data_list) - 1:
                try:
                    response = input("\nPress Enter to continue or 'q' to quit: ").strip().lower()
                    if response == 'q':
                        break
                except KeyboardInterrupt:
                    print("\nStopped by user")
                    break
        
        print(f"\nProcessed {min(i+1, len(data_list))} samples")
        end_time = time.time()
        print(f"Time taken: {end_time - begin_time} seconds")
    except FileNotFoundError:
        print(f"Error: File not found - {args.input}")
        print("Please check the file path or use --input to specify correct path")
        print("Run with --help for usage information")
    except Exception as e:
        print(f"Error: {e}")
