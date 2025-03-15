from typing import Optional, Any

from verl.utils.reward_score.math_v2 import last_boxed_only_string_v2, remove_boxed


def compute_score(solution_str, ground_truth, **argv) -> tuple[float, dict[str, Any]]:
    pred = last_boxed_only_string_v2(solution_str[-100:])
    correct: bool = False
    reward: float = 0
    if pred is None:
        # TODO: return a dict for analysis
        # extra_info = {"acc": correct, "format/answer": "answer:" in solution_str.lower(), "format/boxed": "boxed{" in solution_str, "format/option": False}
        return reward

    pred = remove_boxed(pred)
    if pred.upper() == ground_truth:
        correct = True
        reward = 1
    
    # TODO: return a dict for analysis
    # extra_info = {"acc": correct, "format/answer": "answer:" in solution_str.lower(), "format/boxed": "boxed{" in solution_str, "format/option": pred.upper() in ["A", "B", "C", "D"]}

    return reward


if __name__ == "__main__":
    pass
