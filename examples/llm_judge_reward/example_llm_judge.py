#!/usr/bin/env python3
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
Standalone example for testing LLM-as-Judge reward scoring.

This script demonstrates how to use the LLM-as-Judge reward function
to evaluate generated responses.

Usage:
    # Set your API key as environment variable
    export OPENAI_API_KEY="sk-..."

    # Run with OpenAI
    python example_llm_judge.py --api-url "https://api.openai.com/v1/chat/completions" \
                            --model gpt-4 \
                            --question "What is 7 * 8?" \
                            --response "The answer is 56." \
                            --reference "56"

    # Run with Azure OpenAI
    python example_llm_judge.py --api-url "https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2023-05-15" \
                            --model gpt-4 \
                            --api-key "$AZURE_OPENAI_KEY" \
                            --question "What is the capital of France?" \
                            --response "Paris is the capital of France." \
                            --reference "Paris"

    # Run with custom prompt template
    python example_llm_judge.py --question "Explain photosynthesis" \
                            --response "Photosynthesis is..." \
                            --reference "The process by which..." \
                            --judge-prompt "Evaluate the following answer on accuracy: Question: {question} Answer: {response} Reference: {reference}"
"""

import argparse
import asyncio
import os
from typing import Any

# Import the LLM-as-Judge reward function
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from verl.utils.reward_score.llm_judge import (
    compute_score,
    compute_score_async,
    batch_compute_scores,
)


# Example prompts for different tasks
EXAMPLES = {
    "math": {
        "question": "What is 15 * 7?",
        "response": "15 * 7 = 105",
        "reference": "105",
    },
    "science": {
        "question": "What is the chemical formula for water?",
        "response": "The chemical formula for water is H2O, which consists of two hydrogen atoms bonded to one oxygen atom.",
        "reference": "H2O",
    },
    "history": {
        "question": "When did World War II end?",
        "response": "World War II ended in 1945, with the surrender of Nazi Germany in May and Japan in September.",
        "reference": "1945",
    },
    "coding": {
        "question": "Write a function to reverse a string in Python.",
        "response": "```python\ndef reverse_string(s):\n    return s[::-1]\n```",
        "reference": "s[::-1]",
    },
    "incorrect": {
        "question": "What is the capital of France?",
        "response": "The capital of France is Berlin.",
        "reference": "Paris",
    },
    "partial": {
        "question": "What are the three primary colors?",
        "response": "The primary colors are red and blue.",
        "reference": "Red, blue, and yellow",
    },
    "excellent": {
        "question": "Explain what a black hole is.",
        "response": "A black hole is a region of spacetime where gravity is so strong that nothing, not even light or other electromagnetic waves, can escape from it. Black holes form when massive stars collapse at the end of their life cycles.",
        "reference": "A region of spacetime where gravity prevents escape",
    },
}


async def run_single_judge(args: argparse.Namespace) -> None:
    """Run a single LLM-as-Judge evaluation."""
    print("=" * 80)
    print("LLM-as-Judge: Single Evaluation")
    print("=" * 80)
    print(f"Question: {args.question}")
    print(f"Response: {args.response}")
    print(f"Reference: {args.reference}")
    print(f"Model: {args.model}")
    print(f"API URL: {args.api_url}")
    print("-" * 80)

    result = await compute_score_async(
        data_source="llm_judge",
        solution_str=args.response,
        ground_truth=args.reference,
        extra_info={"question": args.question},
        api_key=args.api_key,
        base_url=args.api_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
        judge_prompt_template=args.judge_prompt,
    )

    print(f"\nScore: {result['score']:.2f} / 1.0")
    print(f"Raw Response: {result.get('raw_response', '')}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    print("=" * 80)


async def run_batch_eval(args: argparse.Namespace) -> None:
    """Run batch LLM-as-Judge evaluation with examples."""
    print("=" * 80)
    print("LLM-as-Judge: Batch Evaluation with Examples")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"API URL: {args.api_url}")
    print(f"Max Concurrent: {args.max_concurrent}")
    print("-" * 80)

    # Build batch items from examples
    items = []
    for name, example in EXAMPLES.items():
        items.append(
            {
                "solution_str": example["response"],
                "ground_truth": example["reference"],
                "extra_info": {"question": example["question"]},
            }
        )

    # Compute scores in batch
    results = await batch_compute_scores(
        items=items,
        api_key=args.api_key,
        base_url=args.api_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        max_concurrent=args.max_concurrent,
        judge_prompt_template=args.judge_prompt,
    )

    # Print results
    total_score = 0.0
    for i, result in enumerate(results):
        example_name = list(EXAMPLES.keys())[i]
        score = result.get("score", 0.0)
        total_score += score
        status_color = "\033[92m" if score >= 0.75 else "\033[93m" if score >= 0.5 else "\033[91m"
        reset_color = "\033[0m"

        print(f"\n[{example_name}]")
        print(f"  Question: {result['question']}")
        print(f"  Score: {status_color}{score:.2f}{reset_color}")
        if result.get("raw_response"):
            print(f"  Judge Reasoning: {result.get('raw_response', '')[:100]}...")
        if result.get("error"):
            print(f"  Error: {result['error']}")

    avg_score = total_score / len(results) if results else 0.0
    print("-" * 80)
    print(f"Average Score: {avg_score:.2f} / 1.0")
    print("=" * 80)


def main() -> None:
    """Main entry point for the example script."""
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge reward scoring example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", os.getenv("AZURE_OPENAI_KEY", "")),
        help="API key for the LLM service (default: OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="https://api.openai.com/v1/chat/completions",
        help="Base URL for the LLM API endpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model name to use for judgment (default: gpt-4)",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question/prompt to evaluate",
    )
    parser.add_argument(
        "--response",
        type=str,
        help="Generated response to evaluate",
    )
    parser.add_argument(
        "--reference",
        type=str,
        help="Reference/ground truth answer",
    )
    parser.add_argument(
        "--judge-prompt",
        type=str,
        help="Custom judge prompt template with {question}, {reference}, {response} placeholders",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens for judge response (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Run in batch mode with example questions",
    )

    args = parser.parse_args()

    # Validate API key
    if not args.api_key:
        print("Error: API key is required. Set OPENAI_API_KEY environment variable or use --api-key.")
        print("\nExample:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  python example_llm_judge.py --batch-mode")
        sys.exit(1)

    # Run in batch or single mode
    if args.batch_mode:
        asyncio.run(run_batch_eval(args))
    else:
        # Check for required arguments in single mode
        if not all([args.question, args.response, args.reference]):
            parser.error(
                "--question, --response, and --reference are required in single mode. "
                "Use --batch-mode to run with example questions."
            )
        asyncio.run(run_single_judge(args))


if __name__ == "__main__":
    main()
