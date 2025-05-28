import json
import os
import hashlib
import random
import glob
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained("/home/share/reasoning/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("/home/yangkai/models/Qwen2.5-32B")

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

def format_question(question_text:str):
    instruction = "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."
    question_text += instruction
    return question_text.strip()

def process_livecode(data_path:str, out_dir:str):
    total = 0
    filtered = 0
    input_file_name_list = ['test.jsonl', 'test2.jsonl', 'test3.jsonl', 'test4.jsonl', 'test5.jsonl', 'test6.jsonl']
    output_file_name = 'livecode_filtered.jsonl'
    seen_questions = set()
    all_items = []
    for input_file_name in input_file_name_list:
        with open(os.path.join(data_path, input_file_name), 'r', encoding='utf-8') as infile:
            for line in tqdm(infile):
                line = line.strip()
                total += 1
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = data.get("question_content").strip()
                    title = data.get("question_title")
                    raw_test_cases = data.get("public_test_cases", "[]")
                    raw_test_cases = json.loads(raw_test_cases)
                    
                    # Skip if token length exceeds limits
                    if not check_token_lengths(question):
                        continue

                    test_cases = {
                        "inputs": [tc["input"] for tc in raw_test_cases],
                        "outputs": [tc["output"] for tc in raw_test_cases]
                    }
                    difficulty = data.get("difficulty")


                    standardized_item = {
                        "question": question,
                        "test_cases": test_cases,
                        "extra_params":{
                            "title": title,
                            "difficulty": difficulty,
                            "source": "LiveCodeBench"
                        }
                    }
                    # Get hash for deduplication
                    question_hash = get_question_hash(question)
                    
                    # Add if not a duplicate
                    if question_hash not in seen_questions:
                        seen_questions.add(question_hash)
                        all_items.append(standardized_item)
                        filtered += 1

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping line due to error: {e}")

    with open(os.path.join(out_dir, output_file_name), 'w', encoding='utf-8') as outfile:
        for item in all_items:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total valid entries processed: {total}, and filtered {filtered}")
    return all_items

def process_codeforces(data_path:str, out_dir:str):
    input_file_name_list = ['train-00000-of-00009.parquet', 'train-00001-of-00009.parquet', 
                            'train-00002-of-00009.parquet', 'train-00003-of-00009.parquet', 
                            'train-00004-of-00009.parquet', 'train-00005-of-00009.parquet',
                            'train-00006-of-00009.parquet','train-00007-of-00009.parquet',
                            'train-00008-of-00009.parquet']
    output_file_name = 'codeforces_filtered.jsonl'

    seen_questions = set()
    all_items = []
    total = 0
    filtered = 0
    print("Loading all parquet files...")
    all_dfs = [pd.read_parquet(os.path.join(data_path, fname)) for fname in input_file_name_list]
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing codeforces"):
        total += 1
        try:
            question = row.get("prompt", "").strip()
            question = format_question(question)
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

            # Skip if token length exceeds limits
            if not check_token_lengths(question):
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
            # Get hash for deduplication
            question_hash = get_question_hash(question)

            # Add if not a duplicate
            if question_hash not in seen_questions:
                seen_questions.add(question_hash)
                all_items.append(standardized_item)
                filtered += 1

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping line due to error: {e}")

    with open(os.path.join(out_dir, output_file_name), 'w', encoding='utf-8') as outfile:
        for item in all_items:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total valid entries processed: {total}, and filtered {filtered}")
    return all_items

def process_apps(data_path:str, out_dir:str):
    total = 0
    filtered = 0
    input_file_name_list = ['train.jsonl']  # , 'test.jsonl'
    output_file_name = 'apps_filtered.jsonl'
    seen_questions = set()
    all_items = []
    for input_file_name in input_file_name_list:
        with open(os.path.join(data_path, input_file_name), 'r', encoding='utf-8') as infile:
            for line in tqdm(infile):
                line = line.strip()
                total += 1
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = data.get("question").strip()
                    title = ""
                    raw_test_cases = data.get("input_output", "[]")
                    if not raw_test_cases:
                        continue
                    raw_test_cases = json.loads(raw_test_cases)
                    # Skip if token length exceeds limits
                    if not check_token_lengths(question):
                        continue

                    test_cases = {
                        "inputs": raw_test_cases['inputs'],
                        "outputs": raw_test_cases['outputs']
                    }
                    difficulty = data.get("difficulty")


                    standardized_item = {
                        "question": question,
                        "test_cases": test_cases,
                        "extra_params":{
                            "title": title,
                            "difficulty": difficulty,
                            "source": "apps"
                        }
                    }
                    # Get hash for deduplication
                    question_hash = get_question_hash(question)
                    
                    # Add if not a duplicate
                    if question_hash not in seen_questions:
                        seen_questions.add(question_hash)
                        all_items.append(standardized_item)
                        filtered += 1

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Skipping line due to error: {e}")
                    print(f">>> Raw line: {repr(line)}")

    with open(os.path.join(out_dir, output_file_name), 'w', encoding='utf-8') as outfile:
        for item in all_items:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total valid entries processed: {total}, and filtered {filtered}")
    return all_items

def process_code_contests(code_contests_data_path:str, out_dir:str):
    input_file_name_list = glob.glob(os.path.join(code_contests_data_path, 'train-*-of-00039-*.parquet'))
    output_file_name = 'code_contests_filtered.jsonl'

    seen_questions = set()
    all_items = []
    total = 0
    filtered = 0
    print("Loading all parquet files...")
    all_dfs = [pd.read_parquet(fname) for fname in input_file_name_list]
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows loaded: {len(df)}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing codecontests"):
        total += 1
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
                print("1111")
                continue

            # Add example
            # examples_text = "\n\nExamples\n\n"
            # for idx, _ in enumerate(raw_example_cases["input"]):
            #     input_text = raw_example_cases["input"][idx].strip()
            #     output_text = raw_example_cases["output"][idx].strip()
            #     examples_text += f"Input\n\n{input_text}\n\nOutput\n\n{output_text}\n\n"
            
            # # 拼接到 question
            # question += examples_text
            question = format_question(question)
            if "Examples" not in question:
                continue

            # Skip if token length exceeds limits
            if not check_token_lengths(question):
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
            # Get hash for deduplication
            question_hash = get_question_hash(question)
            
            # Add if not a duplicate
            if question_hash not in seen_questions:
                seen_questions.add(question_hash)
                all_items.append(standardized_item)
                filtered += 1

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping line due to error: {e}")

    with open(os.path.join(out_dir, output_file_name), 'w', encoding='utf-8') as outfile:
        for item in all_items:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total valid entries processed: {total}, and filtered {filtered}")
    return all_items

def main():
    # Path for the final processed data
    out_dir = "/home/yangkai/data/data_process"
    # data_dir = "/home/share/reasoning/raw_data"
    outfile_name = "raw_merged_code_data.jsonl"

    # livecode_data_path = "/home/yangkai/data/code/code_generation_lite"
    codeforces_data_path = "/home/yangkai/data/code/Codeforces-Python-Submissions/data"
    # apps_data_path = "/home/yangkai/data/code/apps"
    code_contests_data_path = "/home/yangkai/data/code/code_contests/data"

    all_items = []
    # all_items += process_livecode(livecode_data_path, out_dir)
    all_items += process_codeforces(codeforces_data_path, out_dir)
    # all_items += process_apps(apps_data_path, out_dir)
    all_items += process_code_contests(code_contests_data_path, out_dir)

    random.seed(2025)
    random.shuffle(all_items)

    with open(os.path.join(out_dir, outfile_name), 'w', encoding='utf-8') as outfile:
        for item in all_items:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")


main()