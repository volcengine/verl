import json
import os
import argparse
import time
import random
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import concurrent.futures
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Global variables to store model and tokenizer
model = None
tokenizer = None

def load_local_model(model_path: str):
    global model, tokenizer
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    model.to("cuda")
    model.eval()
    

# Function to load and sample math problems
def load_and_sample_math_problems(file_path: str, sample_per_group: int = 10) -> List[Dict[str, Any]]:
    problems = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                level = data.get("extra_params", {}).get("level")
                source = data.get("extra_params", {}).get("source")
                if level in [1, 2, 3] and source is not None:
                    problems.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON")

    print(f"Loaded {len(problems)} valid problems from {file_path}")

    # Group problems by (source, level)
    grouped = {}
    for p in problems:
        level = p["extra_params"]["level"]
        source = p["extra_params"]["source"]
        key = (source, level)
        grouped.setdefault(key, []).append(p)

    sampled = []
    for (source, level), group in grouped.items():
        if len(group) >= sample_per_group:
            sampled_group = random.sample(group, sample_per_group)
        else:
            print(f"Warning: Not enough problems for source={source}, level={level}, using {len(group)} available")
            sampled_group = group
        sampled.extend(sampled_group)

    print(f"Sampled a total of {len(sampled)} problems from all (source, level) combinations.")
    return sampled

def evaluate_answer(response: str, ground_truth: str) -> int:
    from eval_compute_score import compute_score
    
    return compute_score(response, ground_truth)

def query_local_model(prompt: str, params: Dict[str, Any]) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_tokens = params.get("max_tokens", 8000)
    temperature = params.get("temperature", 0.7)
    top_p = params.get("top_p", 0.95)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

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
    parser.add_argument("--input", type=str, default="/home/yangkai/data/data_process/final_merged_math_data.jsonl", 
                        help="Path to math problems")
    parser.add_argument("--output", type=str, default="/home/yangkai/data/data_process/sampled_problems_with_level.jsonl",
                        help="Path to save problems with accuracy")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=8000,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--sample-size", type=int, default=256,
                        help="Number of problems to sample")
    parser.add_argument("--model", type=str, default="/home/yangkai/models/DeepSeek-R1-Distill-Qwen-7B",
                        help="Path to the model directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous results")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Maximum number of concurrent workers")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    problems = load_and_sample_math_problems(args.input, sample_per_group=10)

    # load_local_model(args.model)
    
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
    # for problem in tqdm(problems, desc="Evaluating problems"):
    #     try:
    #         result = process_problem(problem, vllm_params)
    #         results.append(result)
    #     except Exception as e:
    #         print(f"Error processing problem: {e}")
    
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
    
    # Calculate accuracy by (source, level)
    source_level_stats = {}

    for r in results:
        source = r["source"]
        level = r["level"]
        key = (source, level)
        if key not in source_level_stats:
            source_level_stats[key] = {"correct": 0, "total": 0}
        source_level_stats[key]["total"] += 1
        if r["correct"]:
            source_level_stats[key]["correct"] += 1

    # Print detailed stats
    print("\nAccuracy by source and level:")
    for (source, level), stats in sorted(source_level_stats.items()):
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"Source: {source}, Level: {level}, Accuracy: {stats['correct']}/{stats['total']} ({accuracy:.2%})")


if __name__ == "__main__":
    main()