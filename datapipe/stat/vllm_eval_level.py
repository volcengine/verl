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
from datasets import load_dataset
import hashlib

# data_dir = "/home/share/reasoning/raw_data"
data_dir = "/home/yangkai/data/data_process"
orz_math_path = os.path.join(data_dir, "orz_math_13k_collection_hard.json")
additional_dataset_path = os.path.join(data_dir, "hard_problems_with_rate.jsonl")


def is_valid_float(s):
    try:
        val = float(s)
        return True
    except ValueError:
        return False

def get_question_hash(question):
    # Normalize the question by removing extra whitespace and converting to lowercase
    normalized_question = ' '.join(question.strip().lower().split())
    return hashlib.md5(normalized_question.encode('utf-8')).hexdigest()


# Function to load and sample math problems
def load_and_sample_math_problems(file_name: str, sample_per_group: int = 20) -> List[Dict[str, Any]]:
    problems = []
    count = 0
    seen_questions = set()

    if file_name == "orz_math":
        with open(orz_math_path, 'r', encoding='utf-8') as infile:
            orz_data = json.load(infile)
            orz_count = len(orz_data)
            
            for item in tqdm(orz_data):
                if count >= sample_per_group:
                    break
                
                # Extract question from human message
                if len(item) >= 2 and isinstance(item[0], dict) and item[0].get("from") == "human":
                    question = item[0].get("value", "")
                    
                    # Extract answer from assistant message
                    answer = ""
                    if len(item) >= 2 and isinstance(item[1], dict) and item[1].get("from") == "assistant":
                        if "ground_truth" in item[1] and isinstance(item[1]["ground_truth"], dict):
                            answer = item[1]["ground_truth"].get("value", "")

                    question  = question.strip()
                    
                    # Create standardized item
                    if not question:
                        continue
                    if not is_valid_float(answer):
                        continue
                    standardized_item = {
                        "question": question,
                        "answer": answer,
                        "extra_params": {
                            "level": 3,
                            "source": "orz_math_data"
                        }
                    }
                    question_hash = get_question_hash(question)
                    if question_hash not in seen_questions:
                        seen_questions.add(question_hash)
                        problems.append(standardized_item)
                        count += 1
    elif file_name == "general_reasoning":
        level_counts = {i: 0 for i in range(10)}
        with open(additional_dataset_path, 'r', encoding='utf-8') as infile:
            for line in tqdm(infile):
                item = json.loads(line)
                
                # Extract question and reference_answer, adjusting field names if necessary
                question = item.get("question", "")
                if not question:
                    question = item.get("problem", "")  # Alternative field name
                    
                answer = item.get("reference_answer", "")
                if not answer:
                    answer = item.get("answer", "")  # Alternative field name
                
                if not question:
                    continue

                if not is_valid_float(answer):
                    continue
                # Get accuracy/solve rate if available
                accuracy = item.get("accuracy", None) or item.get("solve_rate", None)
                
                if accuracy is None:
                    continue

                level=int(accuracy*10) % 10
                if level_counts[level] >= sample_per_group:
                    continue
                
                question = question.strip()
                    
                # Create consistent format with extra_params
                standardized_item = {
                    "question": question,
                    "answer": answer,
                    "extra_params": {
                        "level": level,
                        "source": "general_reasoning",
                    }
                }
                
                # Get hash for deduplication
                question_hash = get_question_hash(question)
                
                # Add if not a duplicate
                if question_hash not in seen_questions:
                    seen_questions.add(question_hash)
                    problems.append(standardized_item)
                    level_counts[level] += 1
                
                if all(count >= sample_per_group for count in level_counts.values()):
                    break
    elif file_name == "big_math":
        big_math_ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified")
        level_counts = {i: 0 for i in range(10)}
        for split in big_math_ds.keys():
            print(f"Processing Big-Math-RL split: {split}")
            for item in tqdm(big_math_ds[split]):
                # Extract relevant fields
                problem = item.get("problem", "")
                answer = item.get("answer", "")
                solution = item.get("solution", "")
                solve_rate = item.get("llama8b_solve_rate")
                
                # Skip if the item doesn't have llama8b_solve_rate or it's not <= 0.4
                if solve_rate is None:
                    continue
                
                problem = problem.strip()

                level=int(solve_rate*10) % 10
                if level_counts[level] >= sample_per_group:
                    continue

                if not is_valid_float(answer):
                    continue
                # Create standardized item format with extra_params
                standardized_item = {
                    "question": problem,
                    "answer": answer,
                    "extra_params": {
                        "level": level,
                        "source": "big_math_rl_verified",
                    }
                }
                    
                # Get hash for deduplication
                question_hash = get_question_hash(problem)
                
                # Add if not a duplicate
                if question_hash not in seen_questions:
                    seen_questions.add(question_hash)
                    problems.append(standardized_item)
                    level_counts[level] += 1
                
                if all(count >= sample_per_group for count in level_counts.values()):
                    break
    elif file_name == "dapo_math":
        dapo_math_ds = load_dataset("qgallouedec/DAPO-Math-17k-Processed-Scored")
        level_counts = {i: 0 for i in range(10)}
        for split in dapo_math_ds.keys():
            print(f"Processing DAPO-Math split: {split}")
            for item in tqdm(dapo_math_ds[split]):
                # Extract relevant fields
                prompt = item.get("prompt", "")
                solution = item.get("solution", "")  # This is the answer
                data_source = item.get("data_source", "")
                solve_rate = item.get("Qwen3-32B_solve_rate")
                
                # Skip if the item doesn't have Qwen3-32B_solve_rate or it's > 0.5
                if solve_rate is None:
                    continue
                
                # Format question with instructions
                question = prompt.strip()
                
                level=int(solve_rate*10) % 10
                if level_counts[level] >= sample_per_group:
                    continue
                
                if not is_valid_float(solution):
                    continue
                
                # Create standardized item format with extra_params
                standardized_item = {
                    "question": question,
                    "answer": solution,
                    "extra_params": {
                        "level": level,
                        "source": data_source,
                    }
                }
                
                # Get hash for deduplication
                question_hash = get_question_hash(prompt)  # Use original prompt for hashing
                
                # Add if not a duplicate
                if question_hash not in seen_questions:
                    seen_questions.add(question_hash)
                    problems.append(standardized_item)
                    level_counts[level] += 1
                
                if all(count >= sample_per_group for count in level_counts.values()):
                    break
    elif file_name == "skywork":
        level_counts = {i: 0 for i in range(17)}
        skywork_math_ds = load_dataset("Skywork/Skywork-OR1-RL-Data")
        for split in skywork_math_ds.keys():
            if split != "math":
                continue

            print(f"Processing Skywork Math split: {split}")
            for item in tqdm(skywork_math_ds[split]):
                
                # Extract relevant fields
                question = item.get("prompt", "")[0]["content"]
                raw_answer = item.get("reward_model", {}).get("ground_truth", "")
                parsed_answer = json.loads(raw_answer)
                # 提取第一个答案
                if parsed_answer:
                    answer = parsed_answer[0]
                else: 
                    continue
                model_difficulty = item.get("extra_info", None)["model_difficulty"]["DeepSeek-R1-Distill-Qwen-7B"]
                 
                if model_difficulty is None:
                    continue
                
                # Determine level based on solve rate
                level = model_difficulty
                if level_counts[level] >= sample_per_group:
                    continue
                if not is_valid_float(answer):
                        continue
                # Create standardized item format with extra_params
                standardized_item = {
                    "question": question.strip(),
                    "answer": answer,
                    "extra_params": {
                        "level": level,
                        "source": "skywork_math",
                    }
                }
                
                # Get hash for deduplication - use the original question text for hashing
                question_hash = get_question_hash(question)
                
                # Add if not a duplicate
                if question_hash not in seen_questions:
                    seen_questions.add(question_hash)
                    problems.append(standardized_item)
                    level_counts[level] += 1
                
                if all(count >= sample_per_group for count in level_counts.values()):
                    break
    return problems

def evaluate_answer(response: str, ground_truth: str) -> int:
    from eval_compute_score import compute_score
    
    return compute_score(response, ground_truth)

# Function to send requests to vLLM service
def query_vllm(prompt: str, params: Dict[str, Any]) -> str:
    client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1")

    temperature = params.get("temperature", 0.6)
    max_tokens = params.get("max_tokens", 8000)
    top_p = params.get("top_p", 0.95)
    model = params.get("model", "")
    system_prompt = "Solve the problem step by step and present the final answer using \\boxed{...} format. "
    
    try:
        completion = client.completions.create(
            model=model,
            prompt=system_prompt + prompt,
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

    failure = 0
    responses = []
    
    for _ in range(16):
        response = query_vllm(question, vllm_params)
        score = evaluate_answer(response, reference_answer)
        if score != 1:
            failure += 1
        responses.append({
            "response": response,
            "correct": score == 1
        })
    
    return {
        "question": question,
        "reference_answer": reference_answer,
        "model_responses": responses,
        "score": failure,
        "level": level,
        "source": source
    }

def main():
    parser = argparse.ArgumentParser(description="Sample math problems and evaluate pass@1 accuracy")
    parser.add_argument("--input", type=str, default="/home/yangkai/data/data_process/final_merged_math_data.jsonl", 
                        help="Path to math problems")
    parser.add_argument("--dataset", type=str, default="orz_math", 
                        help="Name for datasets")
    parser.add_argument("--output", type=str, default="/home/yangkai/data/data_process/sampled_problems_with_level.jsonl",
                        help="Path to save problems with accuracy")
    parser.add_argument("--temperature", type=float, default=0.6,
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
    
    problems = load_and_sample_math_problems(args.dataset, sample_per_group=20)

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

    with open(args.output, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    if os.path.exists(temp_results_path):
        os.remove(temp_results_path)
    
    # Calculate and print statistics
    total_failures = sum(r["score"] for r in results)
    total_count = len(results) * 16
    mean_failure = total_failures / total_count if total_count > 0 else 0
    
    print(f"\nEvaluation complete.")
    print(f"Overall accuracy: {total_failures}/{total_count} ({mean_failure:.2%})")
    
    # Calculate accuracy by (source, level)
    source_level_stats = {}

    for r in results:
        source = r["source"]
        level = r["level"]
        key = (source, level)
        if key not in source_level_stats:
            source_level_stats[key] = {"score": 0, "total": 0}
        source_level_stats[key]["total"] += 16
        source_level_stats[key]["score"] += r["score"]

    # Print detailed stats
    print("\nAccuracy by source and level:")
    for (source, level), stats in sorted(source_level_stats.items()):
        failure = stats["score"] / stats["total"] if stats["total"] > 0 else 0
        print(f"Source: {source}, Level: {level}, Accuracy: {stats['score']}/{stats['total']} ({failure:.2%})")


if __name__ == "__main__":
    main()