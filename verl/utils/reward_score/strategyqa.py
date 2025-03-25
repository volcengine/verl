import re
from typing import List


def parse_generation(prediction: str) -> List[str]:
    """parse the generated texts to extract the final answer.

    Args:
        prediction (str): generated text

    Returns:
        List[str]: parsed texts
    """
    regex_list = [
        r"<answer>(.*?)</answer>",
        r"<answer>\n(.*?)\n</answer>",
        r"<answer>\\n(.*?)\\n</answer>",
    ]
    parsed_answers = []
    parsed_answer = ""
    for regex in regex_list:
        match = re.search(regex, prediction, re.DOTALL)
        if match and match.group(1):
            parsed_answer = match.group(1)
            break
        parsed_answers.append(parsed_answer.strip())

    return parsed_answers


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think><answer>.*</answer>", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = parse_generation(predict_str)[0]
    return 1.0 if answer.lower() == ground_truth.lower() else 0.0


def compute_score(prompt: str, predict_str: str, ground_truth: str) -> float:
    return 0.9 * acc_reward(predict_str, ground_truth) + 0.1 * format_reward(
        predict_str
    )
