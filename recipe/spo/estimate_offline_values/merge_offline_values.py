# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 SPO authors
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
import argparse
import concurrent.futures
import glob
import json
import os
import random
from collections import defaultdict


def load_and_parse(file_path):
    """
    Reads a JSONL file where each line is a JSON object, and returns a list of parsed objects.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of parsed JSON objects
    """
    try:
        with open(file_path) as file:
            data = [json.loads(line) for line in file]
        print(f"Successfully loaded {len(data)} items from {file_path}")
        return data
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []


def merge_offline_values(
    input_dir, output_file, pattern="offline_value_estimation_subset_*/validation_data/0.jsonl", max_scores_per_prompt=8
):
    """
    Merge offline value estimates from multiple subset directories.

    Args:
        input_dir: Directory containing all subset outputs
        output_file: Path to save the merged offline values JSON file
        pattern: Glob pattern to match subset result files
        max_scores_per_prompt: Maximum number of scores to keep per prompt (default: 8)
    """
    # Find all subset dump files
    search_pattern = os.path.join(input_dir, pattern)
    subset_files = glob.glob(search_pattern)

    if not subset_files:
        print(f"Warning: No files found matching pattern: {search_pattern}")
        return

    print(f"Found {len(subset_files)} subset dump files:")
    for f in sorted(subset_files):
        print(f"  - {f}")

    # Load all subset data using concurrent processing
    all_subset_data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(load_and_parse, f): f for f in subset_files}

        for future in concurrent.futures.as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                result = future.result()
                all_subset_data.extend(result)
            except Exception as exc:
                print(f"{file_name} generated an exception: {exc}")

    print(f"\nTotal items loaded: {len(all_subset_data)}")

    # Merge scores by prompt
    merged_prompt_to_scores = defaultdict(list)
    for item in all_subset_data:
        # Extract the prompt/question from the input field
        # This assumes the format: "...user\n<prompt>\nassistant..."
        try:
            key = item["input"].split("user\n")[-1].split("\nassistant")[0].strip()
            merged_prompt_to_scores[key].append(item["score"])
        except (KeyError, IndexError) as e:
            print(f"Warning: Failed to parse item: {e}")
            continue

    merged_prompts = list(merged_prompt_to_scores.keys())
    print(f"Merged into {len(merged_prompts)} unique prompts")

    # Subsample scores if more than max_scores_per_prompt
    num_prompts_exceeding_max = 0
    for prompt, scores in merged_prompt_to_scores.items():
        if len(scores) > max_scores_per_prompt:
            num_prompts_exceeding_max += 1
            # Randomly sample max_scores_per_prompt scores
            merged_prompt_to_scores[prompt] = random.sample(scores, max_scores_per_prompt)

    if num_prompts_exceeding_max > 0:
        print(
            f"\nSubsampling: {num_prompts_exceeding_max} prompts had more than {max_scores_per_prompt} scores and were randomly subsampled to {max_scores_per_prompt}"
        )

    # Save merged results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(merged_prompt_to_scores, f, indent=2)

    print(f"\nMerged offline values saved to: {output_file}")

    # Print statistics
    score_counts = [len(scores) for scores in merged_prompt_to_scores.values()]
    score_sums = [sum(scores) for scores in merged_prompt_to_scores.values()]

    if score_counts:
        print("\nStatistics (Score Counts per Prompt):")
        print(f"  - Min scores per prompt: {min(score_counts)}")
        print(f"  - Max scores per prompt: {max(score_counts)}")
        print(f"  - Avg scores per prompt: {sum(score_counts) / len(score_counts):.2f}")
        print(f"  - Prompts with >{max_scores_per_prompt} scores (before subsampling): {num_prompts_exceeding_max}")

    if score_sums:
        print("\nStatistics (Sum of Scores per Prompt):")
        print(f"  - Min sum of scores: {min(score_sums):.4f}")
        print(f"  - Max sum of scores: {max(score_sums):.4f}")
        print(f"  - Avg sum of scores: {sum(score_sums) / len(score_sums):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge offline value estimates from multiple subsets into a single file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing all subset outputs (e.g., the trainer.validation_data_dir)",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to save the merged offline values JSON file"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="offline_value_estimation_subset_*/validation_data/0.jsonl",
        help="Glob pattern to match subset result files (default: offline_value_estimation_subset_*/validation_data/0.jsonl)",
    )
    parser.add_argument(
        "--max_scores_per_prompt",
        type=int,
        default=8,
        help="Maximum number of scores to keep per prompt. If a prompt has more scores, randomly subsample to this number (default: 8)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Merging Offline Value Estimates")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output_file}")
    print(f"File pattern: {args.pattern}")
    print(f"Max scores per prompt: {args.max_scores_per_prompt}")
    print("=" * 80 + "\n")

    merge_offline_values(args.input_dir, args.output_file, args.pattern, args.max_scores_per_prompt)


if __name__ == "__main__":
    main()
