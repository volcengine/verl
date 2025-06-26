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
Production Data Loader for Atropos-VERL Integration
This module provides production-ready data loading for RL training with Atropos.

Supports:
- Real datasets (GSM8K, MATH, etc.)
- Parquet files with proper VERL format
- HuggingFace datasets
- Custom data sources
"""

import os
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

from verl.utils.dataset.rl_dataset import RLHFDataset


class AtroposDataLoader:
    """
    Production data loader for Atropos training.

    This loader follows VERL's data patterns and can load from:
    1. Parquet files (production format)
    2. HuggingFace datasets
    3. Custom data sources
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_source = config.get("data_source", "gsm8k")
        self.max_prompts = config.get("max_prompts", 100)
        self.prompt_format = config.get("prompt_format", "chat")

    def load_from_parquet(self, file_path: str) -> List[str]:
        """
        Load prompts from VERL-formatted parquet file.

        This is the production method used in VERL training.
        """
        try:
            df = pd.read_parquet(file_path)
            prompts = []

            for _, row in df.iterrows():
                if self.prompt_format == "chat":
                    # Handle chat format prompts
                    if isinstance(row["prompt"], list):
                        # Extract user content from chat format
                        for message in row["prompt"]:
                            if message.get("role") == "user":
                                prompts.append(message["content"])
                                break
                    else:
                        prompts.append(str(row["prompt"]))
                else:
                    # Handle simple string prompts
                    prompts.append(str(row["prompt"]))

                if len(prompts) >= self.max_prompts:
                    break

            print(f"✓ Loaded {len(prompts)} prompts from parquet: {file_path}")
            return prompts

        except Exception as e:
            print(f"❌ Error loading from parquet {file_path}: {e}")
            return []

    def load_from_huggingface(self, dataset_name: str, split: str = "train") -> List[str]:
        """
        Load prompts from HuggingFace dataset.

        Supports common RLHF datasets like GSM8K, MATH, etc.
        """
        try:
            dataset = load_dataset(dataset_name, split=split)
            prompts = []

            for i, example in enumerate(dataset):
                if i >= self.max_prompts:
                    break

                # Format based on dataset type
                if dataset_name == "gsm8k":
                    prompt = f"Solve this math problem step by step:\n\n{example['question']}\n\nLet's work through this step by step:"
                elif dataset_name == "math":
                    prompt = f"Solve this mathematics problem:\n\n{example['problem']}\n\nProvide a step-by-step solution:"
                elif dataset_name == "hellaswag":
                    prompt = f"Complete the following sentence:\n\n{example['ctx_a']} {example['ctx_b'].capitalize()}"
                else:
                    # Generic format
                    if "question" in example:
                        prompt = example["question"]
                    elif "prompt" in example:
                        prompt = example["prompt"]
                    else:
                        # Use first text field found
                        text_fields = [k for k, v in example.items() if isinstance(v, str) and len(v) > 10]
                        if text_fields:
                            prompt = example[text_fields[0]]
                        else:
                            continue

                prompts.append(prompt)

            print(f"✓ Loaded {len(prompts)} prompts from HuggingFace dataset: {dataset_name}")
            return prompts

        except Exception as e:
            print(f"❌ Error loading from HuggingFace {dataset_name}: {e}")
            return []

    def load_production_prompts(self) -> List[str]:
        """
        Load production prompts using the best available method.

        Priority:
        1. Parquet files (production format)
        2. HuggingFace datasets
        3. Fallback to realistic examples
        """
        prompts = []

        # Try parquet files first (production format)
        parquet_paths = self.config.get("parquet_paths", [])
        for path in parquet_paths:
            if os.path.exists(path):
                prompts = self.load_from_parquet(path)
                if prompts:
                    return prompts

        # Try HuggingFace datasets
        hf_datasets = self.config.get("hf_datasets", ["gsm8k", "math"])
        for dataset_name in hf_datasets:
            prompts = self.load_from_huggingface(dataset_name)
            if prompts:
                return prompts

        # Fallback to realistic production examples
        print("⚠ No production datasets available, using realistic examples...")
        return self._get_realistic_examples()

    def _get_realistic_examples(self) -> List[str]:
        """
        Generate realistic production-style prompts.

        These are examples of what real RLHF prompts look like.
        """
        production_prompts = [
            # Math reasoning (GSM8K style)
            "Janet's dogs eat 2 pounds of food each day. How many pounds of food do her dogs eat in a week? Let's solve this step by step.",
            "A store sells shirts for $25 each and pants for $40 each. If a customer buys 3 shirts and 2 pairs of pants, what is the total cost? Show your work.",
            # Code generation
            "Write a Python function that takes a list of integers and returns the sum of all even numbers. Include proper error handling and docstring.",
            "Implement a binary search algorithm in Python. The function should return the index of the target element or -1 if not found.",
            # Factual QA
            "What are the main differences between supervised and unsupervised learning in machine learning? Provide specific examples for each.",
            "Explain the concept of overfitting in machine learning. What are some techniques to prevent it?",
            # Creative writing
            "Write a short story (2-3 paragraphs) about a robot learning to understand human emotions. Focus on character development and emotional growth.",
            # Analysis task
            "Analyze the pros and cons of using renewable energy sources versus fossil fuels. Consider economic, environmental, and social factors.",
            # Instruction following
            "Given a list of numbers [3, 7, 2, 9, 1, 8], sort them in descending order and explain your sorting method step by step.",
            # Problem solving
            "A company wants to reduce its carbon footprint by 30% in the next 5 years. What are three specific strategies they could implement? Explain the expected impact of each.",
            # Technical explanation
            "Explain how a transformer model processes sequential data, including the role of attention mechanisms and positional encoding.",
            # Logical reasoning
            "If all roses are flowers and some flowers are red, can we conclude that some roses are red? Explain your reasoning.",
            # Data analysis
            "Given a dataset of student test scores, explain how you would identify outliers and what methods you would use to handle them.",
            # System design
            "Design a simple recommendation system for an e-commerce website. Explain the key components and how they would work together.",
            # Ethics and safety
            "What are the potential risks and benefits of deploying large language models in healthcare applications? Consider privacy, accuracy, and accessibility.",
        ]

        # Return subset based on max_prompts
        return production_prompts[: self.max_prompts]


def create_verl_rl_dataset(prompts: List[str], config: Dict[str, Any]) -> RLHFDataset:
    """
    Create a VERL RL dataset from prompts.

    This converts prompts into the proper VERL format for RL training.
    """
    # Convert prompts to VERL format
    data = []
    for i, prompt in enumerate(prompts):
        # Create VERL-compatible data structure
        example = {
            "data_source": config.get("data_source", "atropos"),
            "prompt": [{"role": "user", "content": prompt}],
            "ability": config.get("ability", "general"),
            "reward_model": {
                "style": "model",  # or "rule" for rule-based rewards
                "ground_truth": None,  # Will be provided by Atropos environments
            },
            "extra_info": {
                "index": i,
                "split": "train",
                "need_tools_kwargs": False,
            },
        }
        data.append(example)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Create VERL RL dataset
    dataset = RLHFDataset(
        parquet_files=[df],  # Pass DataFrame directly
        tokenizer=None,  # Will be set by trainer
        max_prompt_length=config.get("max_prompt_length", 512),
        max_response_length=config.get("max_response_length", 512),
        prompt_key="prompt",
        reward_fn_key="data_source",
        return_raw_input_ids=False,
        return_raw_chat=False,
        return_full_prompt=False,
        truncation="error",
        need_tools_kwargs=False,
    )

    return dataset


# Example usage
if __name__ == "__main__":
    # Production configuration
    config = {
        "data_source": "atropos_integration",
        "max_prompts": 50,
        "prompt_format": "chat",
        "parquet_paths": ["~/data/rlhf/gsm8k/train.parquet", "~/data/rlhf/math/train.parquet"],
        "hf_datasets": ["gsm8k", "math", "hellaswag"],
        "max_prompt_length": 512,
        "max_response_length": 512,
        "ability": "general",
    }

    # Load production prompts
    loader = AtroposDataLoader(config)
    prompts = loader.load_production_prompts()

    print(f"Loaded {len(prompts)} production prompts")
    print("Sample prompts:")
    for i, prompt in enumerate(prompts[:3]):
        print(f"{i + 1}. {prompt[:100]}...")
