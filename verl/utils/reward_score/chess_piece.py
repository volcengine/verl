import re
import random

def extract_solution(solution_str):

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def compute_score_train(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    
    answer = extract_solution(solution_str=solution_str)
    
    do_print = random.randint(1, 64) == 1
    if do_print:
        print("-------------------------")
        print(f"Predicted: {answer} | GT: {ground_truth['piece_name']}")
        print(f"Solution: {solution_str}") 
    
    if answer is None:
        return 0
    else:
        if answer == ground_truth['piece_name']:
            return score
        else:
            return format_score

def compute_score_test(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    
    answer = extract_solution(solution_str=solution_str)
    
    do_print = random.randint(1, 64) == 1
    if do_print:
        print("-------------------------")
        print(f"Predicted: {answer} | GT: {ground_truth['piece_name']}")
        print(f"Solution: {solution_str}") 

    if answer == ground_truth['piece_name']:
        return score
    else:
        return 0
    
    
    