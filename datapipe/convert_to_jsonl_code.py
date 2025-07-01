import json
import os
import hashlib
import random
import glob
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Dict, Any
from datetime import datetime
import pickle
import zlib
import base64
import time
import re

print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained("/home/share/reasoning/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("/home/share/reasoning/DeepSeek-R1-Distill-Qwen-7B")

def get_question_hash(question):
    # Normalize the question by removing extra whitespace and converting to lowercase
    normalized_question = ' '.join(question.strip().lower().split())
    return hashlib.md5(normalized_question.encode('utf-8')).hexdigest()

# Function to check token length using the specified tokenizer
def check_token_lengths(question, max_question_tokens=2048):
    if not question:
        return False
    
    # Check question length
    question_encoding = tokenizer(question, return_attention_mask=False)
    question_tokens = len(question_encoding.input_ids)
    if question_tokens > max_question_tokens:
        return False

    return True

def process_livecode(data_path:str, start_date:str, end_date:str, is_train:bool):
    total = 0
    train_input_list = ['test.jsonl', 'test2.jsonl', 'test3.jsonl', 'test4.jsonl', 'test5.jsonl', 'test6.jsonl']
    benchmark_input_list = ['test5.jsonl']
    all_items = []
    input_file_name_list = train_input_list if is_train else benchmark_input_list
    for input_file_name in input_file_name_list:
        with open(os.path.join(data_path, input_file_name), 'r', encoding='utf-8') as infile:
            for line in tqdm(infile):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    contest_date = datetime.fromisoformat(data.get("contest_date"))
                    if start_date is not None:
                        p_start_date = datetime.strptime(start_date, '%Y-%m-%d')
                        if p_start_date > contest_date:
                            continue
                    if end_date is not None:
                        p_end_date = datetime.strptime(end_date, '%Y-%m-%d')
                        if p_end_date < contest_date:
                            continue

                    question = data.get("question_content").strip()
                    title = data.get("question_title")
                    public_test_cases = data.get("public_test_cases", "[]")
                    public_test_cases = json.loads(public_test_cases)

                    private_test_cases = data.get("private_test_cases", "[]")
                    try:
                        private_test_cases = json.loads(private_test_cases)
                    except Exception as e:
                        private_test_cases = json.loads(
                            pickle.loads(zlib.decompress(base64.b64decode(private_test_cases.encode('utf-8'))  # type: ignore
                                                    )))  # type: ignore

                    raw_test_cases = public_test_cases + private_test_cases


                    metadata = json.loads(data.get("metadata", "{}"))
                    if metadata:
                        fn_name = metadata.get("func_name", None)
                    else:
                        fn_name = None
                    
                    test_cases = {
                        "inputs": [tc["input"] for tc in raw_test_cases],
                        "outputs": [tc["output"] for tc in raw_test_cases],
                        "fn_name": fn_name
                    }
                    difficulty = data.get("difficulty")
                    starter_code = data.get("starter_code", None)


                    standardized_item = {
                        "question": question,
                        "test_cases": test_cases,
                        "extra_params":{
                            "title": title,
                            "difficulty": difficulty,
                            "source": "LiveCodeBench",
                            "starter_code": starter_code
                        }
                    }
                    total += 1
                    all_items.append(standardized_item)

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping line due to error: {e}")

    print(f"Total valid entries processed for livecode: {total}")
    return all_items

def process_codeforces(data_path:str):
    input_file_name_list = ['train-00000-of-00009.parquet', 'train-00001-of-00009.parquet', 
                            'train-00002-of-00009.parquet', 'train-00003-of-00009.parquet', 
                            'train-00004-of-00009.parquet', 'train-00005-of-00009.parquet',
                            'train-00006-of-00009.parquet','train-00007-of-00009.parquet',
                            'train-00008-of-00009.parquet']

    all_items = []
    total = 0
    print("Loading all parquet files...")
    all_dfs = [pd.read_parquet(os.path.join(data_path, fname)) for fname in input_file_name_list]
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing codeforces"):
        try:
            question = row.get("prompt", "").strip()
            title = row.get("name", "").strip()
            raw_test_cases = row.get("test_cases", "[]")
            # raw_test_cases = json.loads(raw_test_cases)
            test_cases = {
                    "inputs": [tc["input"] for tc in raw_test_cases],
                    "outputs": [tc["output"] for tc in raw_test_cases]
                }
            difficulty = str(row.get("rating", "")).strip()

            if not test_cases["inputs"]:
                continue

            # 检查 inputs 是否为 list 且其中每个元素都是 str
            if not all(isinstance(i, str) for i in test_cases["inputs"]):
                continue

            standardized_item = {
                    "question": question,
                    "test_cases": test_cases,
                    "extra_params":{
                        "title": title,
                        "difficulty": difficulty,
                        "source": "codeforces"
                    }
                }
            total += 1
            all_items.append(standardized_item)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping line due to error: {e}")

    print(f"Total valid entries processed for codeforces: {total}")
    return all_items

def process_code_contests(code_contests_data_path:str):
    input_file_name_list = glob.glob(os.path.join(code_contests_data_path, 'train-*-of-00039-*.parquet'))

    all_items = []
    total = 0
    print("Loading all parquet files...")
    all_dfs = [pd.read_parquet(fname) for fname in input_file_name_list]
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing codecontests"):
        try:
            question = row.get("description", "").strip()
            title = row.get("name", "").strip()
            raw_example_cases = row.get("public_tests", {"input": [], "output": []})
            raw_test_cases = row.get("private_tests", {"input": [], "output": []})
            test_cases = {
                    "inputs": list(raw_example_cases["input"]) + list(raw_test_cases["input"]),
                    "outputs": list(raw_example_cases["output"]) + list(raw_test_cases["output"])
                }
            difficulty = str(row.get("cf_rating", "")).strip()

            if not test_cases["inputs"]:
                continue

            # # 检查 inputs 是否为 list 且其中每个元素都是 str
            if not all(isinstance(i, str) for i in test_cases["inputs"]):
                continue
            
            if "Examples" not in question:
                continue

            standardized_item = {
                    "question": question,
                    "test_cases": test_cases,
                    "extra_params":{
                        "title": title,
                        "difficulty": difficulty,
                        "source": "code_contests"
                    }
                }
            total += 1
            all_items.append(standardized_item)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping line due to error: {e}")

    print(f"Total valid entries processed for code contests: {total}")
    return all_items

def filter_train_code_data(all_items: List[Dict[str, Any]], max_test_cases: int = 2000, min_test_cases: int = 5) -> List[Dict[str, Any]]:
    multiple_answers_patterns = [
        r"multiple possible answers",
        r"multiple answers",
        r"more than one answer",
        r"any valid answer",
        r"any correct answer",
        r"any possible answer",
        r"any valid solution",
        r"any correct solution"
    ]
    
    # Compile regex patterns
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in multiple_answers_patterns]
    
    # Add patterns for input/output specifications
    input_spec_patterns = [
        r"Input Specification:",
        r"Input\n",
    ]
    output_spec_patterns = [
        r"Output Specification:",
        r"Output\n",
    ]
    
    input_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in input_spec_patterns]
    output_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in output_spec_patterns]
    
    filtered_samples = []
    seen_questions = set()
    total_samples = len(all_items)
    multiple_answers_filtered_count = 0
    test_cases_filtered_count = 0
    none_filtered_count = 0
    overflow_filtered_count = 0
    duplicate_filtered_count = 0
    spec_filtered_count = 0  # Counter for samples filtered due to missing specifications
    img_filtered_count = 0

    for sample in all_items:
        try:
            question_hash = get_question_hash(sample["question"])
            if question_hash in seen_questions:
                duplicate_filtered_count += 1
                continue
            seen_questions.add(question_hash)

            if not check_token_lengths(sample["question"]):
                overflow_filtered_count += 1
                continue

            data_source = sample.get("extra_params", {}).get("source", "")
            if data_source == "LiveCodeBench":
                filtered_samples.append(sample)
                continue

            question = sample.get("question", "").lower()

            if "<img" in question:
                img_filtered_count += 1
                continue

            has_multiple_answers = any(pattern.search(question) for pattern in patterns)
            test_cases = sample.get("test_cases", {})
            num_test_cases = len(test_cases.get("inputs", []))

            if "None" in sample.get("test_cases", {}).get("inputs", []):
                none_filtered_count += 1
                continue
            
            if has_multiple_answers:
                multiple_answers_filtered_count += 1
                continue
                
            if num_test_cases > max_test_cases or num_test_cases < min_test_cases:
                test_cases_filtered_count += 1
                continue    
            
            # Check for input/output specifications
            has_input_spec = any(pattern.search(question) for pattern in input_patterns)
            has_output_spec = any(pattern.search(question) for pattern in output_patterns)
            
            if not (has_input_spec and has_output_spec):
                spec_filtered_count += 1
                continue
            
            filtered_samples.append(sample)
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    print(f"postprocessing complete:")
    print(f"Total samples: {total_samples}")
    # print(f"Overflow filtered samples: {overflow_filtered_count}")
    # print(f"Duplicate filtered samples: {duplicate_filtered_count}")
    # print(f"Multiple answers filtered samples: {multiple_answers_filtered_count}")
    # print(f"Test cases filtered samples: {test_cases_filtered_count}")
    # print(f"None filtered samples: {none_filtered_count}")
    # print(f"Missing specifications filtered samples: {spec_filtered_count}")
    # print(f"Img filtered samples: {img_filtered_count}")
    # print(f"Remaining samples: {len(filtered_samples)}")
    return filtered_samples 

def filter_benchmark_code_data(all_items:List[Dict[str, Any]]):
    filtered_items = []
    for item in all_items:
        if not check_token_lengths(item["question"]):
            continue
        filtered_items.append(item)
    return filtered_items

def main():
    start_time = time.time()
    # Path for the final processed data
    out_dir = "/home/liunazhou/data"
    train_outfile = "rl_code_train_0701_test.jsonl"
    benchmark_outfile = "rl_code_benchmark_0701_test.jsonl"

    livecode_data_path = "/home/yangkai/data/code/code_generation_lite"
    codeforces_data_path = "/home/yangkai/data/code/Codeforces-Python-Submissions/data"
    code_contests_data_path = "/home/yangkai/data/code/code_contests/data"

    train_items = []
    train_items += process_livecode(livecode_data_path, start_date="2023-05-01", end_date="2024-09-30", is_train=True)
    train_items += process_livecode(livecode_data_path, start_date="2025-02-01", end_date="2025-05-02", is_train=True)
    train_items += process_codeforces(codeforces_data_path)
    train_items += process_code_contests(code_contests_data_path)
    train_items = filter_train_code_data(train_items)

    random.seed(2025)
    random.shuffle(train_items)

    benchmark_items = []
    benchmark_items += process_livecode(livecode_data_path, start_date="2024-10-01", end_date="2025-01-31", is_train=False)
    benchmark_items = filter_benchmark_code_data(benchmark_items)

    with open(os.path.join(out_dir, train_outfile), 'w', encoding='utf-8') as outfile:
        for item in train_items:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, benchmark_outfile), 'w', encoding='utf-8') as outfile:
        for item in benchmark_items:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total train items: {len(train_items)}")
    print(f"Total benchmark items: {len(benchmark_items)}")
    print(f"Time taken: {time.time() - start_time} seconds")
main()