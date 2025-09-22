# Copyright 2025 CollabLLM team and/or its affiliates
# Copyright 2025 Bytedance Ltd. and/or its affiliates

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

#!/usr/bin/env python3
"""
# available datasets: 
# math-hard(-large), medium(-large), bigcodebench(-large)
# to create your own dataset, refer to https://github.com/Wuyxin/collabllm

DATASET=math-hard-large

python recipe/collabllm/process_dataset.py \
  --dataset collabllm/collabllm-multiturn-$DATASET  \
  --local_dir $HOME/data/collabllm-$DATASET \
  --dataset_type sft

python recipe/collabllm/process_dataset.py \
  --dataset collabllm/collabllm-multiturn-$DATASET  \
  --local_dir $HOME/data/collabllm-$DATASET \
  --dataset_type rl
  

Preprocess collabllm/collabllm-multiturn-math-hard into (ground_truth, extra_info).

- ground_truth: picked from --prefer_field (default: single_turn_completion),
                falling back to --fallback_field (default: completion)
- extra_info:   a shallow copy of the original example plus bookkeeping fields
- reward_model: {"style": "rule", "ground_truth": ground_truth}

Saves one parquet per split into --local_dir and a small JSON preview.
"""

import argparse
import json
import os
import uuid
from typing import Any, Optional

from datasets import Dataset, concatenate_datasets, load_dataset

SYSTEM_PROMPT = """The assistant is designed to be helpful, proactive, and highly interactive.

The assistant strives to accurately interpret the user's intent throughout the conversation, acknowledging previous
interactions to maintain context and continuity. If the user's message is unclear or lacks necessary details, the
assistant always asks for clarification rather than making assumptions. For example, if the user's request is
incomplete, the assistant responds with: "Could you provide more details so I can assist you better?"

The assistant asks specific follow-up questions and offers suggestions based on the user's needs, avoiding vague or
generic prompts. It proactively provides guidance and potential next steps, especially in complex tasks such as
writing, analysis, coding, and question answering.

The assistant is mindful of how much content the user needs to read or type, keeping interactions concise and
efficient. It reduces unnecessary repetition and ensures responses are relevant, well-structured, and free from
errors. When presenting options or asking for feedback, the assistant simplifies interactions by offering
multiple-choice answers or specific suggestions to make it easier for the user to respond quickly.

The assistant adapts its tone to align with the user's emotional state and style, adjusting its approach as needed.
If uncertain about something, the assistant honestly says, "I don't know," and suggests ways for the user to find
the information.

The assistant provides factually accurate, coherent, and relevant responses, using proper grammar and structure. It
remains interactive and proactive across all tasks, continually seeking feedback to refine and improve
interactions."""


# Required fields: "prompt", "ground_truth", "extra_info"
# In "extra_info" dict:
# (1) Rquired: "single_turn_prompt", which is the specific problem used to inform the user simulator,
# (2) Optional: "task_desc" (a short task description),
# (3) Optional: other fields for customized reward computation
def collapse_example(example: dict[str, Any]) -> dict[str, Any]:
    if "prompt" not in example:
        raise ValueError("Missing required 'prompt' field.")

    ground_truth = (
        example.get("ground_truth") or example.get("single_turn_completion") or example.get("completion") or ""
    )

    extra_info = {}
    for k, v in example.items():
        if k in ("prompt", "ground_truth", "extra_info"):
            continue
        extra_info.setdefault(k, v)  # keep extra_info values if keys overlap

    # make sure extra_info has the required fields
    assert "single_turn_prompt" in extra_info, "Missing 'single_turn_prompt' in extra_info."

    # add system prompt as the beginning of the list
    example["prompt"] = [{"role": "system", "content": SYSTEM_PROMPT}] + example["prompt"]

    extra_info.setdefault("prompt", example["prompt"])  # save the original prompt
    extra_info.setdefault(
        "interaction_kwargs",
        {
            "name": "collabllm",
            "single_turn_prompt": extra_info.pop("single_turn_prompt"),
            "task_desc": extra_info.pop("task_desc", "general ask-for-assistance task"),
        },
    )
    return {
        "prompt": example["prompt"],
        "ground_truth": ground_truth,
        "raw_prompt": example["prompt"],  # save the original prompt
        "extra_info": extra_info,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "data_source": "collabllm",
        "agent_name": "collabllm_agent",
        "index": str(uuid.uuid4()),
    }


# ---------- IO helpers ----------
def save_parquet(ds_split: Dataset, filename: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{filename}.parquet")
    ds_split.to_parquet(path)
    print(f"[OK] Wrote {filename}.parquet → {path} ({len(ds_split)} rows)")


def maybe_copy_to_hdfs(local_dir: str, hdfs_dir: Optional[str]) -> None:
    if not hdfs_dir:
        return
    try:
        from verl.utils.hdfs_io import copy, makedirs  # type: ignore
    except Exception as e:
        print(f"[WARN] Skipping HDFS copy (verl not available): {e}")
        return
    makedirs(hdfs_dir)
    copy(src=local_dir, dst=hdfs_dir)
    print(f"[OK] Copied {local_dir} → {hdfs_dir}")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", default="collabllm/collabllm-multiturn-math-hard", help="HF dataset path or local dir/file."
    )
    ap.add_argument("--task_desc", default="solving math problems", help="Task description for the dataset.")
    ap.add_argument("--local_dir", default="~/data/collabllm-math-hard", help="Output directory.")
    ap.add_argument("--hdfs_dir", default=None, help="Optional HDFS destination (requires verl).")
    ap.add_argument(
        "--validation_size", type=float, default=0.1, help="Validation split size (fraction or absolute int)."
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    ap.add_argument("--num_proc", type=int, default=1, help="Parallel workers for map().")
    ap.add_argument("--dataset_type", default="rl", choices=["rl", "sft"], help="Type of dataset (e.g., 'rl', 'sft').")
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.local_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading dataset: {args.dataset}")
    ds_dict = load_dataset(args.dataset)
    parts = list(ds_dict.values())
    ds_all: Dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    # Dataset({
    #     features: ['prompt', 'completion', 'conv_id', 'score', 'single_turn_prompt',
    #       'single_turn_completion', 'single_turn_metadata', 'turn_id', 'sessions', 'rewards'],
    #     num_rows: xxx
    # })

    if args.dataset_type == "rl":
        # If multiple splits exist, merge them before collapsing/splitting.
        ds_all = ds_all.map(lambda x: {"task_desc": args.task_desc}, num_proc=args.num_proc)

        print(f"[INFO] Collapsing to formatted fields on {len(ds_all)} rows…")
        ds_all = ds_all.map(
            function=collapse_example,
            remove_columns=ds_all.column_names,
            num_proc=args.num_proc,
        )

        def dedup_by_prompt(dataset):
            seen = set()
            unique_rows = []
            for ex in dataset:
                prompt_key = json.dumps(ex["prompt"], sort_keys=True, ensure_ascii=False)
                if prompt_key not in seen:
                    seen.add(prompt_key)
                    unique_rows.append(ex)
            return Dataset.from_list(unique_rows)

        ds_all = dedup_by_prompt(ds_all)

    elif args.dataset_type == "sft":
        df = ds_all.to_pandas()

        # Sort so that within each conv_id the highest turn_id is first,
        # and if multiple rows share the same turn_id, the highest score comes first
        df = df.sort_values(["conv_id", "turn_id", "score"], ascending=[True, False, False])

        # Keep only the top row per conv_id
        df = df.drop_duplicates(subset="conv_id", keep="first")

        # Back to HF Dataset
        ds_all = Dataset.from_pandas(df, preserve_index=False)

        # Append assistant response into prompt list
        def append_completion(example):
            example["prompt"] = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + example["prompt"]
                + [{"role": "assistant", "content": example["completion"]}]
            )
            return example

        ds_all = ds_all.map(append_completion)

        # Keep only prompt column
        cols_to_remove = [col for col in ds_all.column_names if col != "prompt"]
        ds_all = ds_all.remove_columns(cols_to_remove)

    print(f"[INFO] Splitting with validation_size={args.validation_size}, seed={args.seed}")
    split = ds_all.train_test_split(test_size=args.validation_size, seed=args.seed, shuffle=True)
    train_ds, val_ds = split["train"], split["test"]
    print(train_ds, val_ds)

    save_parquet(train_ds, f"{args.dataset_type}_train", out_dir)
    save_parquet(val_ds, f"{args.dataset_type}_validation", out_dir)

    maybe_copy_to_hdfs(local_dir=out_dir, hdfs_dir=args.hdfs_dir)
    print(f"[DONE] {args.dataset_type}_train.parquet and {args.dataset_type}_validation.parquet written.")


if __name__ == "__main__":
    main()
