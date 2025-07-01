from datasets import Dataset
import json
import ast
import argparse
from typing import Any, List, Dict


def find_arrays_in_string(text: str) -> List[tuple]:
    arrays = []
    i = 0
    while i < len(text):
        if text[i] == '[':
            # Found start of array, find the matching closing bracket
            bracket_count = 1
            j = i + 1
            while j < len(text) and bracket_count > 0:
                if text[j] == '[':
                    bracket_count += 1
                elif text[j] == ']':
                    bracket_count -= 1
                j += 1
            
            if bracket_count == 0:
                # Found complete array
                array_str = text[i:j]
                arrays.append((i, j, array_str))
                i = j
            else:
                i += 1
        else:
            i += 1
    
    return arrays


def is_1d_array(value: str) -> bool:
    if not isinstance(value, str):
        return False
    
    value = value.strip()
    
    if not (value.startswith('[') and value.endswith(']')):
        return False    
    
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            return False
        
        # Check if any element is a list or dict (would make it 2D or nested)
        for item in parsed:
            if isinstance(item, (list, dict)):
                return False
        
        return True
    except (json.JSONDecodeError, ValueError):
        try:
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, list):
                return False
            
            for item in parsed:
                if isinstance(item, (list, dict)):
                    return False
            
            return True
        except (ValueError, SyntaxError):
            return False


def is_2d_array(value: str) -> bool:
    if not isinstance(value, str):
        return False
    
    value = value.strip()
    
    if not (value.startswith('[') and value.endswith(']')):
        return False
    
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            return False
        
        if len(parsed) == 0:
            return False
        
        # Check if inner elements are lists
        for item in parsed:
            if not isinstance(item, list):
                return False
            # Check that inner lists don't contain nested structures
            for inner_item in item:
                if isinstance(inner_item, (list, dict)):
                    return False
        
        return True
    except (json.JSONDecodeError, ValueError):
        try:
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, list):
                return False
            
            if len(parsed) == 0:
                return False
            
            for item in parsed:
                if not isinstance(item, list):
                    return False
                for inner_item in item:
                    if isinstance(inner_item, (list, dict)):
                        return False
            
            return True
        except (ValueError, SyntaxError):
            return False


def convert_string_array_to_space_separated(value: str) -> str:
    if is_1d_array(value):
        try:
            parsed = json.loads(value)
            return ' '.join(str(item) for item in parsed)
        except (json.JSONDecodeError, ValueError):
            try:
                parsed = ast.literal_eval(value)
                return ' '.join(str(item) for item in parsed)
            except (ValueError, SyntaxError):
                return value
    elif is_2d_array(value):
        try:
            parsed = json.loads(value)
            rows = []
            for row in parsed:
                rows.append(' '.join(str(item) for item in row))
            return '\n'.join(rows)
        except (json.JSONDecodeError, ValueError):
            try:
                parsed = ast.literal_eval(value)
                rows = []
                for row in parsed:
                    rows.append(' '.join(str(item) for item in row))
                return '\n'.join(rows)
            except (ValueError, SyntaxError):
                return value
    else:
        return value


def process_reward_model_inputs(inputs_list: List[str]) -> List[str]:
    processed_inputs = []
    
    for input_str in inputs_list:
        if isinstance(input_str, str):
            lines = input_str.split('\n')
            processed_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    processed_lines.append(line)
                    continue
                
                arrays_found = find_arrays_in_string(line)
                
                if arrays_found:
                    processed_line = line
                    for start, end, array_str in reversed(arrays_found):
                        if is_1d_array(array_str) or is_2d_array(array_str):
                            processed_array = convert_string_array_to_space_separated(array_str)
                            processed_line = processed_line[:start] + processed_array + processed_line[end:]
                    
                    processed_lines.append(processed_line)
                else:
                    if is_1d_array(line) or is_2d_array(line):
                        processed_line = convert_string_array_to_space_separated(line)
                    else:
                        processed_line = line
                    
                    processed_lines.append(processed_line)
            
            processed_input = '\n'.join(processed_lines)
            processed_inputs.append(processed_input)
        else:
            processed_inputs.append(input_str)
    
    return processed_inputs


def process_dataset_row(row: Dict[str, Any]) -> Dict[str, Any]:
    processed_row = row.copy()
    
    if 'reward_model' in row and isinstance(row['reward_model'], dict):
        reward_model = row['reward_model'].copy()
        
        if 'ground_truth' in reward_model and isinstance(reward_model['ground_truth'], dict):
            ground_truth = reward_model['ground_truth'].copy()
            
            need_process = ground_truth.get("fn_name", None) is None or "starter code" not in row['prompt'][0]['content'].lower()
            if 'inputs' in ground_truth and isinstance(ground_truth['inputs'], list) and need_process:
                ground_truth['inputs'] = process_reward_model_inputs(ground_truth['inputs'])
                reward_model['ground_truth'] = ground_truth
                processed_row['reward_model'] = reward_model
    
    return processed_row


def process_parquet_dataset(input_path: str, output_path: str = None):
    print(f"Loading dataset from: {input_path}")
    
    dataset = Dataset.from_parquet(input_path)
    print(f"Dataset loaded with {len(dataset)} rows")
    print(f"Features: {dataset.features}")
    
    processed_data = []
    conversion_count = 0
    found_2d_array = False
    print("Processing rows...")
    
    for i, row in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(dataset)} rows")
        
        original_inputs = row['reward_model']['ground_truth']['inputs']
        
        processed_row = process_dataset_row(row)
        
        new_inputs = processed_row['reward_model']['ground_truth']['inputs']
        if original_inputs != new_inputs:
            conversion_count += 1
            if conversion_count == 1:
                print("\nExample of conversion:")
                print("Original input:")
                for line in original_inputs[0].split('\n'):
                    print(line)
                print("\nProcessed input:")
                for line in new_inputs[0].split('\n'):
                    print(line)
        
        processed_data.append(processed_row)
    
    print(f"Total conversions made: {conversion_count}")
    
    processed_dataset = Dataset.from_list(processed_data)
    
    if output_path:
        print(f"Saving processed dataset to: {output_path}")
        processed_dataset.to_parquet(output_path)
        print("Dataset saved successfully!")
    
    return processed_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process parquet dataset to convert string arrays in reward_model inputs")
    parser.add_argument("--input", help="Path to input parquet file", default="/home/share/reasoning/rl_code_train_0627.parquet")
    parser.add_argument("--output", help="Path to output parquet file", default="/home/share/reasoning/rl_code_train_0627_array_filtered.parquet")
    
    args = parser.parse_args()


    process_parquet_dataset(args.input, args.output)
