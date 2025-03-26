import re
import math
from typing import Dict, List, Tuple

from evaluate import load
from thefuzz.fuzz import ratio
from lingua import LanguageDetectorBuilder


def aggregate_rewards(rewards: Dict[str, float]) -> float:
    """Aggregate multiple rewards into a single reward value.

    Args:
        rewards (Dict[str, float]): Dictionary of rewards to aggregate.

    Returns:
        float: Aggregated reward value.
    """
    rewards_sum = 0.0
    for name, value in rewards.items():
        if "reward" in name:
            rewards_sum += value

    return rewards_sum


def compute_meteor(
    preds: List[str],
    labels: List[str],
) -> float:
    """compute the METEOR score of the model predictions.

    Args:
        preds (List[str]): predicted texts
        labels (List[str]): ground-truth texts

    Returns:
        float: METEOR score
    """
    assert len(preds) == len(labels), "preds and labels must have the same length"

    meteor_scorer = load("meteor")
    meteor_score = meteor_scorer.compute(predictions=preds, references=labels).get(
        "meteor", 0.0
    )

    return round(meteor_score, 3)


def grade_generation_length(given_answer: str) -> float:
    if len(given_answer) < 64:
        return -0.5
    elif len(given_answer) < 128:
        return 0.0
    elif len(given_answer) < 256:
        return 0.1
    elif len(given_answer) < 512:
        return 0.2
    return 0.0


def grade_language_monotony(given_answer: str, language: str = "zh") -> bool:
    if language == "zh":
        target_language = "CHINESE"
    elif language == "en":
        target_language = "ENGLISH"
    else:
        raise ValueError(f"Language {language} must be specified correctly.")

    detector = (
        LanguageDetectorBuilder.from_all_languages()
        .with_preloaded_language_models()
        .build()
    )
    confidence_list = detector.compute_language_confidence_values(given_answer)
    lang2conf = {
        confidence.language.name: confidence.value for confidence in confidence_list
    }
    if lang2conf[target_language] < 0.8:
        return False

    return True


def grade_language_repetition(
    given_answer: str,
    language: str = "zh",
    ngram: int = 2,
    tau: float = 1.0,
    steepness: float = 4.0,
) -> float:
    """
    Calculate a smoothed diversity reward based on distinct-n score for the given text,
    with temperature scaling to control the influence of the reward.

    Args:
        given_answer (str): The text to evaluate
        language (str): Language code, default "zh" for Chinese
        ngram (int): Size of n-grams to use, default 2
        tau (float): Temperature parameter in range [0, 1] to control reward scaling, default 1.0
                    - tau = 0: No diversity reward (always returns 0)
                    - tau = 1: Full diversity reward (returns value in [-1, 0])
                    - 0 < tau < 1: Scaled diversity reward

    Returns:
        float: A scaled reward value between -1 and 0, where values closer to 0 indicate higher diversity
    """
    # Ensure tau is in valid range
    tau = max(0.0, min(1.0, tau))

    # If tau is 0, diversity doesn't matter, return 0 reward
    if tau == 0:
        return 0.0

    # Check if input is empty
    if not given_answer or len(given_answer.strip()) == 0:
        return -1.0 * tau  # Minimum reward for empty text, scaled by tau

    # Chinese tokenization
    if language == "zh":
        try:
            import jieba

            tokens = list(jieba.cut(given_answer))
        except ImportError:
            # Fallback: simple character-based tokenization for Chinese
            tokens = list(given_answer)
    else:
        # For other languages, split by whitespace (simple approach)
        tokens = given_answer.split()

    # Generate n-grams
    ngrams = []
    for i in range(len(tokens) - ngram + 1):
        ngrams.append(tuple(tokens[i : i + ngram]))

    # Calculate distinct-n score
    if not ngrams:
        return -1.0 * tau  # Minimum reward if no n-grams could be formed, scaled by tau

    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))

    # Distinct-n score: ratio of unique n-grams to total n-grams
    distinct_n = unique_ngrams / total_ngrams if total_ngrams > 0 else 0

    # Smoothing function to map distinct-n (range 0-1) to reward (range -1 to 0)
    # Using a sigmoid-like function that gives more reward as diversity increases
    # and approaches 0 (max reward) as distinct_n approaches 1

    # Parameters to tune the smoothing function
    steepness = steepness  # Controls how steep the reward curve is
    midpoint = 0.5  # The distinct-n value that gives a reward of -0.5

    # Sigmoid-like function mapped to [-1, 0]
    raw_reward = -1 + 1 / (1 + math.exp(-(math.e**steepness) * (distinct_n - midpoint)))

    # Apply temperature scaling - scales the reward by tau
    scaled_reward = raw_reward * tau

    # Ensure the reward stays within [-1, 0]
    scaled_reward = max(-1, min(0, scaled_reward))

    return scaled_reward


def grade_law_solution_by_outcome(
    pred: str,
    label: str,
    enable_soft_match: bool = True,
    enable_fuzzy_match: bool = False,
) -> bool:
    """compute the accuracy of the model predictions.

    Args:
        pred (str): predicted text
        label (str): ground-truth text
        enable_soft_match (bool, optional): enable soft match. Defaults to False.
        enable_fuzzy_match (bool, optional): enable fuzzy match. Defaults to True.

    Returns:
        float: accuracy
    """

    def is_digit_equal(pred: str, label: str) -> bool:
        pred_digit = (
            (
                pred.replace("年", "")
                .replace("月", "")
                .replace("日", "")
                .replace("亿", "")
            )
            .replace("千万", "")
            .replace("百万", "")
            .replace("十万", "")
            .replace("万", "")
            .replace("千", "")
            .replace("百", "")
        )
        label_digit = (
            (
                label.replace("年", "")
                .replace("月", "")
                .replace("日", "")
                .replace("亿", "")
            )
            .replace("千万", "")
            .replace("百万", "")
            .replace("十万", "")
            .replace("万", "")
            .replace("千", "")
            .replace("百", "")
        )
        return pred_digit == label_digit

    def is_soft_match(
        pred: str, label: str, sample_type: str, float_range: int = 3
    ) -> bool:
        if sample_type == "量刑":
            both_month_unit = "月" in pred and "月" in label
            pred = pred.replace("年", "").replace("月", "").replace("日", "")
            label = label.replace("年", "").replace("月", "").replace("日", "")
            if pred.isdigit() and label.isdigit():
                label_int = int(label)
                pred_int = int(pred)
                label_upper = min(label_int + float_range, 12)
                label_lower = max(label_int - float_range, 0)
                # NOTE: no need to consider the strings like 3年10个月 vs. 4年1个月 in LawGPT
                if label_lower <= pred_int <= label_upper:
                    return True
                elif both_month_unit and label_int == pred_int:
                    return True
                else:
                    return False
        return False

    if not label or not pred:
        return False

    if label == "" or pred == "":
        return False

    # exact match trial
    sample_type = "量刑" if "[刑期]" in label else "罚金"

    parsed_pred = parse_generation([pred])[0]
    parsed_label = parse_generation([label])[0]
    # print(f"parsed_pred: {parsed_pred}, parsed_label: {parsed_label}")

    if parsed_pred == "" or parsed_label == "":
        # empty prediction
        return False
    elif parsed_pred == parsed_label or is_digit_equal(parsed_pred, parsed_label):
        # exact match trial
        return True
    elif enable_fuzzy_match:
        # fuzzy match trial (if enabled)
        if ratio(parsed_pred, parsed_label) > 90:
            return True
    elif enable_soft_match:
        # soft match trial (if enabled)
        if is_soft_match(parsed_pred, parsed_label, sample_type):
            return True

    return False


def grade_law_solution_by_process(
    given_answers: List[str], ground_truths: List[str]
) -> bool:
    return False


def parse_generation(
    preds: List[str],
) -> List[str]:
    """parse the generated texts to extract the final answer.

    Args:
        preds (List[str]): generated texts

    Returns:
        List[str]: parsed texts
    """
    regex_list = [
        r"<answer>\[刑期\](.*?)</answer>",
        r"<answer>\[金额\](.*?)</answer>",
        r"<answer>[\\n]*\[刑期\](.*?)[\\n]*</answer>"
        r"<answer>[\\n]*\[金额\](.*?)[\\n]*</answer>",
    ]
    parsed_answers = []
    for pred in preds:
        parsed_answer = ""
        for regex in regex_list:
            match = re.findall(regex, pred)
            if len(match) > 0:
                # pick the last match as the parsed answer
                parsed_answer = match[-1]
                break
        if parsed_answer.strip() == "":
            # if no match found, use the whole text as the parsed answer
            parsed_answer = (
                pred.rsplit("<answer>", 1)[-1]
                .rsplit("</answer>", 1)[0]
                .strip("[刑期]")
                .strip("[金额]")
            )
        parsed_answers.append(parsed_answer.strip())

    return parsed_answers


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = (
                final_answer.split("#### ")[1].replace(",", "").replace("$", "")
            )
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def validate_answer_format(passage: str) -> bool:
    if "[刑期]" not in passage and "[金额]" not in passage:
        return False
    if "<think>" not in passage or "</think>" not in passage:
        return False
    if "<answer>" not in passage or "</answer>" not in passage:
        return False
    return True


def compute_score(prompt, solution_str, ground_truth) -> Tuple[float, Dict[str, float]]:
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    solution_str = (
        solution_str.rsplit("<|im_end|>", 1)[-1]
        .split("assistant", 1)[-1]
        .rsplit("<|endoftext|>", 1)[0]
        .strip()
    )  # output
    predicted_answer = parse_generation([solution_str])[0]
    reference_answer = parse_generation([ground_truth])[0]

    eval_result = {
        "input": prompt,
        "output": solution_str,
        "reference": ground_truth,
        "predicted_answer": predicted_answer,
        "reference_answer": reference_answer,
        # "meteor": meteor_score,
        "format_rewards": 0,
        "length_rewards": 0,
        "unk_error_rewards": 0,
        "repetition_rewards": 0,
        "language_monotony_rewards": 0,
        "correctness_rewards": 0,
        "soft_exact_match": 0,
        "hard_exact_match": 0,
    }
    model_answer = solution_str

    # Step 0. Check if the model response is valid
    if validate_answer_format(solution_str) is False:
        eval_result["format_rewards"] = 0.
        # if "<think>" in solution_str and "</think>" in solution_str:
            # eval_result["format_rewards"] = 0.0
        # elif "<think>" in solution_str or "</think>" in solution_str:
            # eval_result["format_rewards"] = -0.5
        # elif "<answer>" in solution_str and "</answer>" in solution_str:
            # eval_result["format_rewards"] = 0.0
        # elif "<answer>" in solution_str or "</answer>" in solution_str:
            # eval_result["format_rewards"] = -0.5
        # else:
            # eval_result["format_rewards"] = -0.5

    # Check if the model response is too long or too short
    # eval_result["length_rewards"] = grade_generation_length(solution_str)
    eval_result["length_rewards"] = 0.

    # Step 1. check the language monotony / repetition reward of the model response
    language_monotony_score = grade_language_monotony(solution_str, language="zh")
    if not language_monotony_score:
        # eval_result["language_monotony_rewards"] = -0.5
        eval_result["language_monotony_rewards"] = 0.

    language_repetition_score = grade_language_repetition(
        solution_str, language="zh", ngram=1, tau=1.0, steepness=4.0
    )
    if language_repetition_score < -0.5:
        # eval_result["repetition_rewards"] = language_repetition_score
        eval_result["repetition_rewards"] = 0.

    # Step 1. extract the answer from the model response
    if (
        predicted_answer is None
        or predicted_answer == ""
        or reference_answer is None
        or reference_answer == ""
        or solution_str.count("<think>") != 1
        or solution_str.count("</think>") != 1
        or solution_str.count("<answer>") != 1
        or solution_str.count("</answer>") != 1
        or (solution_str.count("[刑期]") != 1 and solution_str.count("[金额]") != 1)
    ):
        # eval_result["format_rewards"] = -0.5
        eval_result["format_rewards"] = 0.

    # Step 2. Process the ground truth(s)
    ground_truth = reference_answer
    # keywords for process supervision
    # _ = (
    # input.ground_truth.get("keywords", None)
    # if isinstance(input.ground_truth, dict)
    # else None
    # )
    if ground_truth is None:
        eval_result["unk_error_rewards"] = -0.5

    # Step 3. Convert single answer to list for uniform processing
    if isinstance(ground_truth, (str, float, int)):
        ground_truths = [ground_truth]

    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        processed_ground_truths.append(truth)

    if not processed_ground_truths:
        eval_result["unk_error_rewards"] = -0.5

    # Step 4. Check if the answer is correct against all possible correct answers
    # (Add float range for soft match: Penalty: +/- 3 month, Money: +/- 1000 RMB)
    for ground_truth in processed_ground_truths:
        is_soft_correct = grade_law_solution_by_outcome(
            model_answer,
            ground_truth,
            enable_soft_match=True,
            enable_fuzzy_match=False,
        ) or grade_law_solution_by_process(model_answer, ground_truth)
        if is_soft_correct:
            eval_result["correctness_rewards"] = 1.0
            eval_result["soft_exact_match"] += 1.0
            is_hard_correct = grade_law_solution_by_outcome(
                model_answer,
                ground_truth,
                enable_soft_match=False,
                enable_fuzzy_match=False,
            ) or grade_law_solution_by_process(model_answer, ground_truth)
            if is_hard_correct:
                eval_result["hard_exact_match"] += 1.0

    # Step 5. If all else fails, assign incorrect reward and return
    if eval_result["correctness_rewards"] == 0:
        eval_result["correctness_rewards"] = -1.0
    elif eval_result["correctness_rewards"] == 1:
        # pass
        # set all other rewards to 0 if the answer is correct
        eval_result["format_rewards"] = 0
        eval_result["length_rewards"] = 0
        eval_result["unk_error_rewards"] = 0
        eval_result["repetition_rewards"] = 0
        eval_result["language_monotony_rewards"] = 0

    # Step 7. Aggregate rewards and return
    reward = aggregate_rewards(eval_result)

    # return 0.9 * acc_reward(predict_str, ground_truth) + 0.1 * format_reward(predict_str)
    return reward, eval_result
