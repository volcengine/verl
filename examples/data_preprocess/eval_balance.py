import json
import os
import argparse
import time
import random
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import concurrent.futures
from collections import defaultdict


# Function to load and sample math problems
def load_and_sample_math_problems(file_path: str, sample_size_per_group: int = 30) -> List[Dict[str, Any]]:
    grouped_problems = defaultdict(list)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                source = data["extra_params"].get("source", "unknown")
                level = data["extra_params"].get("level", "unknown")
                if level in [1, 2, 3]:  # Only use levels 1, 2, 3
                    grouped_problems[(source, level)].append(data)
            except json.JSONDecodeError:
                print("Warning: Could not parse line as JSON")

    sampled_problems = []
    for (source, level), problems in grouped_problems.items():
        count = len(problems)
        if count >= sample_size_per_group:
            sampled = random.sample(problems, sample_size_per_group)
        else:
            sampled = problems  # if less than sample size
            print(f"Warning: Only {count} problems for source={source}, level={level}")
        sampled_problems.extend(sampled)

    print(f"Total sampled problems: {len(sampled_problems)}")
    return sampled_problems


def evaluate_answer(response: str, ground_truth: str) -> int:
    from eval_compute_score import compute_score
    
    return compute_score(response, ground_truth)


# Function to send requests to vLLM service
def query_vllm(prompt: str, params: Dict[str, Any]) -> str:
    client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1")

    temperature = params.get("temperature", 0.7)
    max_tokens = params.get("max_tokens", 8000)
    top_p = params.get("top_p", 0.95)
    model = params.get("model", "")
    
    try:
        completion = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        return completion.choices[0].text
    except Exception as e:
        print(f"Request failed: {e}")
        return ""


# Function to process a single problem (for concurrent execution)
def process_problem(problem: Dict[str, Any], vllm_params: Dict[str, Any]) -> Dict[str, Any]:
    question = problem["question"]
    reference_answer = problem["answer"]
    level = problem["extra_params"].get("level", "unknown")
    source = problem["extra_params"].get("source", "unknown")
    
    response = query_vllm(question, vllm_params)
    
    score = evaluate_answer(response, reference_answer)
    
    return {
        "question": question,
        "reference_answer": reference_answer,
        "model_response": response,
        "correct": score == 1,
        "level": level,
        "source": source
    }


def main():
    parser = argparse.ArgumentParser(description="Sample math problems and evaluate pass@1 accuracy")
    parser.add_argument("--input", type=str, default="/home/share/reasoning/rl_math_data.jsonl", 
                        help="Path to math problems")
    parser.add_argument("--output", type=str, default="/home/share/reasoning/sampled_problems_with_accuracy.jsonl",
                        help="Path to save problems with accuracy")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=10000,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--sample-size", type=int, default=256,
                        help="Number of problems to sample")
    parser.add_argument("--model", type=str, default="/home/share/reasoning/Qwen2.5-32B-Instruct",
                        help="Path to the model directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous results")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Maximum number of concurrent workers")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    problems = load_and_sample_math_problems(args.input, sample_size_per_group=20)
    
    vllm_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": 0.95,
        "model": args.model
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    results = []
    
    temp_results_path = args.output + ".temp"

    if args.resume and os.path.exists(temp_results_path):
        print(f"Resuming from {temp_results_path}...")
        results = []
        processed_questions = set()
        
        with open(temp_results_path, 'r', encoding='utf-8') as f:
            for line in f:
                problem = json.loads(line.strip())
                results.append(problem)
                processed_questions.add(problem["question"])
        
        print(f"Loaded {len(results)} already processed problems")
        
        problems = [p for p in problems if p["question"] not in processed_questions]
        print(f"Remaining {len(problems)} problems to process")
    
    # Process problems in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_problem = {
            executor.submit(process_problem, problem, vllm_params): problem 
            for problem in problems
        }
        
        for future in tqdm(
            concurrent.futures.as_completed(future_to_problem), 
            total=len(problems),
            desc="Evaluating problems"
        ):
            try:
                result = future.result()
                results.append(result)
                
                # Write results so far to temporary file
                with open(temp_results_path, 'w', encoding='utf-8') as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                        
            except Exception as e:
                print(f"Error processing problem: {e}")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    if os.path.exists(temp_results_path):
        os.remove(temp_results_path)
    
    # Calculate and print statistics
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\nEvaluation complete.")
    print(f"Overall accuracy: {correct_count}/{total_count} ({accuracy:.2%})")
    
    # Calculate accuracy by level
    level_stats = {}
    for r in results:
        level = r["level"]
        if level not in level_stats:
            level_stats[level] = {"correct": 0, "total": 0}
        level_stats[level]["total"] += 1
        if r["correct"]:
            level_stats[level]["correct"] += 1
    
    print("\nAccuracy by level:")
    for level, stats in sorted(level_stats.items()):
        level_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"Level {level}: {stats['correct']}/{stats['total']} ({level_accuracy:.2%})")
    
    # Calculate accuracy by source
    source_stats = {}
    for r in results:
        source = r["source"]
        if source not in source_stats:
            source_stats[source] = {"correct": 0, "total": 0}
        source_stats[source]["total"] += 1
        if r["correct"]:
            source_stats[source]["correct"] += 1
    
    print("\nAccuracy by source:")
    for source, stats in sorted(source_stats.items()):
        source_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{source}: {stats['correct']}/{stats['total']} ({source_accuracy:.2%})")
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
