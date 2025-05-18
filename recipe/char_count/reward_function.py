"""
Reward function
"""

from verl.utils.reward_score import math


def char_count_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    try:
        last_boxed_string = math.last_boxed_only_string(solution_str)
        if last_boxed_string is None:
            return 0
        solution = math.remove_boxed(last_boxed_string)
        if solution == ground_truth:
            return 1
        else:
            return 0
    except:
        print(ground_truth, solution_str)
        return 0
