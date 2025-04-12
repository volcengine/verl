import re
import math 
def are_floats_equivalent(a, b, tolerance=1e-9):
    return math.isclose(a, b, abs_tol=tolerance)

def is_float_or_int(s):
    # Define the regex pattern for a float or int number
    number_pattern = r'^[-+]?\d+$|^[-+]?\d*\.\d+$|^[-+]?\d+\.\d*$'

    # Match the string with the pattern
    if re.match(number_pattern, s):
        return True
    else:
        return False

def is_number(s):
    pattern = r'^[-+]?\d*\.?\d+(e[-+]?\d+)?$'
    return bool(re.match(pattern, s))

def ratio_to_float(ratio_str):
    """Convert a ratio string like '1:4' to a float value (1/4 = 0.25)."""
    try:
        # Split the string by colon and convert each part to float
        num, denom = map(float, ratio_str.split(':'))
        return num / denom if denom != 0 else None
    except (ValueError, ZeroDivisionError):
        return None

def grade_answer(answer, ground_truth, exclude_year_for_margin=True, percent_relaxation=True, substring_relaxation=True):
    answer = answer.lower().strip()
    ground_truth = ground_truth.lower().strip()
    
    # Remove unwanted characters
    answer = re.sub(r'[\[\]$,]', '', answer).rstrip('.')
    ground_truth = re.sub(r'[\[\]$,]', '', ground_truth).rstrip('.')
    
    # Try numerical comparison
    try:
        golds_float = float(ground_truth)
        predict_float = float(answer)
        margin = abs(golds_float * 0.05)
        
        if abs(predict_float - golds_float) <= margin:
            return True
        
        if exclude_year_for_margin and 1000 < golds_float < 3000 and golds_float.is_integer():
            if not are_floats_equivalent(predict_float, golds_float):
                return False
        
        if percent_relaxation:
            if are_floats_equivalent(predict_float, golds_float * 100) or are_floats_equivalent(predict_float * 100, golds_float):
                return True
    except ValueError:
        pass
    
    # Exact match check
    if answer == ground_truth:
        return True
    
    # Substring relaxation
    if substring_relaxation:
        pred_tokens = answer.split()
        if len(pred_tokens) > 1 and is_number(pred_tokens[0]) and not is_number(pred_tokens[1]):
            if pred_tokens[0] == ground_truth:
                return True
    
    if answer in ground_truth or ground_truth in answer:
        return True
    
    # Percentage handling
    if re.search(r'\d+%', answer):
        answer = answer.split('or')[-1].strip('%').strip()
        if answer == ground_truth:
            return True
    
    # Final numerical check after string operations
    if is_number(answer) and is_number(ground_truth):
        golds_float = float(ground_truth)
        predict_float = float(answer)
        margin = abs(golds_float * 0.05)
        if abs(predict_float - golds_float) <= margin:
            return True
    
    return False


def chartqa_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def chartqa_accuracy_reward(predict_str: str, ground_truth: list) -> float:
    ground_truth = ground_truth[0]
    try:
        ground_truth = ground_truth.strip()
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if grade_answer(given_answer, ground_truth):
            return 1.0
    except Exception:
        pass

    return 0.0


def compute_score(predict_str: str, ground_truth: str) -> float:
    return 0.5 * chartqa_accuracy_reward(predict_str, ground_truth) + 0.5 * chartqa_format_reward(predict_str)