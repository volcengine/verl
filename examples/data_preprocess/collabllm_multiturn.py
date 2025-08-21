#!/usr/bin/env python3
"""
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

from datasets import load_dataset, Dataset
from typing import Any, Dict, Tuple, Optional


SYSTEM_PROMPT = '''The assistant is designed to be helpful, proactive, and highly interactive.

The assistant strives to accurately interpret the user's intent throughout the conversation, acknowledging previous interactions to maintain context and continuity. If the user's message is unclear or lacks necessary details, the assistant always asks for clarification rather than making assumptions. For example, if the user's request is incomplete, the assistant responds with: "Could you provide more details so I can assist you better?"

The assistant asks specific follow-up questions and offers suggestions based on the user's needs, avoiding vague or generic prompts. It proactively provides guidance and potential next steps, especially in complex tasks such as writing, analysis, coding, and question answering.

The assistant is mindful of how much content the user needs to read or type, keeping interactions concise and efficient. It reduces unnecessary repetition and ensures responses are relevant, well-structured, and free from errors. When presenting options or asking for feedback, the assistant simplifies interactions by offering multiple-choice answers or specific suggestions to make it easier for the user to respond quickly.

The assistant adapts its tone to align with the user's emotional state and style, adjusting its approach as needed. If uncertain about something, the assistant honestly says, "I don't know," and suggests ways for the user to find the information.

The assistant provides factually accurate, coherent, and relevant responses, using proper grammar and structure. It remains interactive and proactive across all tasks, continually seeking feedback to refine and improve interactions.'''

# Required fields: "prompt", "ground_truth", "extra_info"
# In "extra_info" dict:
# (1) Rquired: "single_turn_prompt", which is the specific problem used to inform the user simulator, 
# (2) Optional: "task_desc" (a short task description), 
# (3) Optional: other fields for customized reward computation
def collapse_example(example: Dict[str, Any]) -> Dict[str, Any]:
    if "prompt" not in example:
        raise ValueError("Missing required 'prompt' field.")

    ground_truth = (
        example.get("ground_truth")
        or example.get("single_turn_completion")
        or example.get("completion")
        or ""
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

    extra_info.setdefault("prompt", example["prompt"]) # save the original prompt
    extra_info.setdefault("interaction_kwargs", {
                "name": "collabllm",
                "single_turn_prompt": extra_info.pop("single_turn_prompt"),
                "task_desc": extra_info.pop("task_desc", "general assistance task")
    }) 
    return {"prompt": example["prompt"], 
            "ground_truth": ground_truth,
            "raw_prompt": example["prompt"], # save the original prompt
            "extra_info": extra_info, 
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "data_source": 'collabllm'
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
    ap.add_argument("--dataset", default="collabllm/collabllm-multiturn-math-hard",
                    help="HF dataset path or local dir/file.")
    ap.add_argument("--task_desc", default="solving math problems",
                    help="Task description for the dataset.")
    ap.add_argument("--local_dir", default="~/data/collabllm-math-hard",
                    help="Output directory.")
    ap.add_argument("--hdfs_dir", default=None,
                    help="Optional HDFS destination (requires verl).")
    ap.add_argument("--validation_size", type=float, default=0.1,
                    help="Validation split size (fraction or absolute int).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for splitting.")
    ap.add_argument("--num_proc", type=int, default=1,
                    help="Parallel workers for map().")
    ap.add_argument("--dataset_type", default="rl", 
                    choices=["rl", "sft"], 
                    help="Type of dataset (e.g., 'rl', 'sft').")
    args = ap.parse_args()

    out_dir = os.path.expanduser(args.local_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading dataset: {args.dataset}")
    ds_dict = load_dataset(args.dataset)

    # If multiple splits exist, merge them before collapsing/splitting.
    parts = list(ds_dict.values())
    ds_all: Dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    ds_all = ds_all.map(lambda x: {"task_desc": args.task_desc}, num_proc=args.num_proc)

    print(f"[INFO] Collapsing to formatted fields on {len(ds_all)} rows…")
    ds_all = ds_all.map(
        function=collapse_example,
        remove_columns=ds_all.column_names,  
        num_proc=args.num_proc,
    )

    print(f"[INFO] Splitting with validation_size={args.validation_size}, seed={args.seed}")
    split = ds_all.train_test_split(test_size=args.validation_size, seed=args.seed, shuffle=True)
    train_ds, val_ds = split["train"], split["test"]
    print(train_ds, val_ds)
    
    print(train_ds["extra_info"][0].keys())

    save_parquet(train_ds, "train", out_dir)
    save_parquet(val_ds, "validation", out_dir)

    maybe_copy_to_hdfs(local_dir=out_dir, hdfs_dir=args.hdfs_dir)
    print("[DONE] train.parquet and validation.parquet written.")


if __name__ == "__main__":
    main()