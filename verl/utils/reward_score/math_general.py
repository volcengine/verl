# import aiohttp
# import asyncio
import requests
import time
import re
from .grader import grade_answer
from collections import Counter
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from local_verifier import verify_latex_answer

MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", 14000)) - 1024
UNIT_LENGTH = max(MAX_RESPONSE_LENGTH//4 * 3, MAX_RESPONSE_LENGTH-3702)
MAX_LENGTH_CONTROL = int(os.getenv("MAX_LENGTH_CONTROL", UNIT_LENGTH))

url = "https://verifier.yuewu.ml/api"
headers = {
    "Content-Type": "application/json",
    "Authentication": "RCbUvAw8nEv_jAuQa82uvAoZBiUv0fMEc28FUddmh78"  # Replace with your API key if needed.
}

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

_CHAT_TAG = "<|im_start|>assistant<|im_sep|>"

_PRED_CFG = LatexExtractionConfig(
    normalization_config=NormalizationConfig(
        nits=False,
        malformed_operators=False,
        basic_latex=True,
        boxed=True,
        units=True,
    ),
    boxed_match_priority=0,
    try_extract_without_anchor=False,
)
_GOLD_CFG = LatexExtractionConfig()        # vanilla rules


def verify_latex_answer(answer: str, gold: str, kind: str = "math") -> dict:
    """Return {'type': kind, 'LaTeXAgreementScore': float, 'Extracted': (ans, gold)}."""
    if kind == "writing":
        return {"type": "writing"}

    if _CHAT_TAG in answer:                       # strip chat delimiter
        answer = answer.split(_CHAT_TAG, 1)[-1]

    # parse ground-truth, first as $...$, fallback to plain
    gold_parsed = parse(f"${gold}$", extraction_config=[_GOLD_CFG]) or parse(gold)
    if not gold_parsed:
        return {"type": "math", "LaTeXAgreementScore": 0.0, "Extracted": ("N/A", "N/A")}

    # parse prediction with normalisation; fallback to plain
    ans_parsed = parse(answer, extraction_config=[_PRED_CFG]) or parse(answer)
    if not ans_parsed:
        return {"type": "math", "LaTeXAgreementScore": 0.0,
                "Extracted": ("N/A", str(gold_parsed[0]))}

    try:
        score = float(
            verify(ans_parsed[0], gold_parsed[0],
                   cmp_mode=ComparisonMode.NUMERIC_FIRST)  # avoids buggy symbolic branch
        )
    except Exception:      # SymPy or math_verify edge-case
        score = 0.0

    return {"type": "math", "LaTeXAgreementScore": score,
            "Extracted": (answer, gold)}

def compute_acc_reward(solution_str, ground_truth):
    """Returns 1. if the completion is correct, 0. if not."""

    # First, use re to extract <answer>...</answer> from the completion. Note that the regex should handle multi-line strings.
    pattern = r"<answer>.*?</answer>"
    matches = re.findall(pattern, solution_str, re.DOTALL)  # Use re.DOTALL to match across newlines

    extracted_sol = None
    if len(matches) == 1:
        extracted_sol = matches[0] 
    else:
        # return 0. # Edit: This sometimes lead to the actor just giving up and returning empty responses. Need to give correct answers a higher score even if they don't match the regex.

        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            try:
                extracted_sol = string_in_last_boxed
                if extracted_sol.strip() == "\\boxed{}": # If the completion is just \boxed{},
                    return 0.
                # Implement a simple heuristic to check if the completion is correct.
                answer = remove_boxed(string_in_last_boxed)
                if is_equiv(answer, ground_truth):
                    return 1.
                elif answer.isnumeric() and ground_truth.isnumeric():
                    return 0.
            except Exception as e:
                pass

    if extracted_sol is None:
        return 0.

    # ########## flask based verifier ###
    # payload = {
    #     "answer": extracted_sol,
    #     "ground_truth": ground_truth
    # }
    # delay = 0.1   # initial delay (in seconds)
    # max_delay = 5 # maximum delay (in seconds)

    # while True:
    #     try:
    #         response = requests.post(url, headers=headers, json=payload, timeout=300)
    #         resp_json = response.json()
    #         retval = resp_json['LaTeXAgreementScore']
    #         return retval
    #     except requests.exceptions.RequestException as e:
    #         time.sleep(delay)
    #         delay = min(delay * 2, max_delay)
    # ##########
    
    ########## local verifier
    response = verify_latex_answer(extracted_sol, ground_truth)
    return response['LaTeXAgreementScore']
    ##########


def compute_format_reward(solution_str):
    """Returns 0.5 if the completion is in the correct format, 0. if not."""

    # Use re to check if the completion contains \boxed{...}
    if re.search(r"\\boxed{.*?}", solution_str):
        return 0.5
    
    return 0.

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>.*\\boxed\{.*\}.*<\|im_end\|>$"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [1. if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>.*\\boxed\{.*\}.*<\|im_end\|>"
    pattern = r"<think>.*?</think>.*<answer>.*</answer>.*<\|im_end\|>"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [1. if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.25
    if text.count("\n</think>\n") == 1:
        count += 0.25
    if text.count("\n<answer>\n") == 1:
        count += 0.25
    if text.count("\n</answer>") == 1:
        count += 0.25
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def strict_xml(text) -> float:
    # Encourage the patter by parts.

    reward = 0.

    # check if the think tag only occurs once
    pattern = r"<think>.*?</think>"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 1:
        reward += 0.5
    
    # check if the answer tag only occurs once
    pattern = r"\\boxed{.*}.*<\|im_end\|>"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 1:
        reward += 0.5

        # reward -= (len(text.split("</answer>")[-1]) - 1)*0.001

    return reward

def compute_repetition_penalty(text, n_gram_size=5):


    words = re.findall(r'\w+', text.lower())
    if len(words) < n_gram_size:
        return 0.
    
    # Generate n-grams
    # n_grams = [" ".join(words[i:i + n_gram_size]) for i in range(len(words) - n_gram_size + 1)]
    n_grams = list(zip(*[words[i:] for i in range(n_gram_size)]))
    
    # Count occurrences of each n-gram
    n_gram_counts = Counter(n_grams)

    # Calculate repetition score (fraction of repeated n-grams)
    total_n_grams = len(n_grams)

    repeated_n_grams = sum(1 for count in n_gram_counts.values() if count > 5)
    repeated_n_gram_ratio = repeated_n_grams / total_n_grams if total_n_grams > 0 else 0.0

    max_repetition = max(n_gram_counts.values())
    if max_repetition < 5:
        max_repetition_ratio = 0.
    else:
        max_repetition_ratio = max_repetition / (len(words)/n_gram_size)

    repetition_score = max(repeated_n_gram_ratio, max_repetition_ratio)

    return - repetition_score

def compute_score(solution_str, ground_truth, response_length, max_response_length=MAX_RESPONSE_LENGTH):
    """Reward function that checks if the completion is the same as the ground truth."""

    split, ground_truth = ground_truth.split("######")

    tool_count = min(solution_str.count("<tool>"), solution_str.count("</tool>"))

    # Remove the prompt from the completion.
    if "<|im_start|>assistant<|im_sep|>" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant<|im_sep|>")[1]

    # strict_format_reward = strict_format_reward_func([solution_str])[0]
    soft_format_reward = soft_format_reward_func([solution_str])[0]

    xml_reward = count_xml(solution_str)
    # xml_reward = strict_xml(solution_str)
    
    # Remove the end tag from the completion.
    incomplete = "<|im_end|>" not in solution_str
    solution_str = solution_str.replace("<|im_end|>", "").strip()

    # If the split is test, we can directly compare the completion with the ground truth.
    if split == "test":
        # Try to extract \boxed{...} from the completion.
        solution_str = last_boxed_only_string(solution_str)
        if solution_str is not None:
            return compute_acc_reward(solution_str, ground_truth), {"no_wandb_ans": solution_str, "no_wandb_sol": ground_truth}
        else:
            # If the completion does not contain \boxed{...}, return 0.
            return 0., {"no_wandb_ans": solution_str, "no_wandb_sol": ground_truth}

    # if there are more than one think tags, return -1 to prevent reward hacking of regex
    invalid_think = solution_str.count("<think>") > 1 or solution_str.count("</think>") > 1
    no_think = "<think>" not in solution_str or "</think>" not in solution_str
    progress = 0.
    weights = [2., 0.25, 0.5, 0.25, 0.25]

    # Apply repetition penalty
    repetition_penalty_score = compute_repetition_penalty(solution_str)

    tool_reward = 0.
    # set tool reward to 1 if <tool> is used inside <think> tag
    if "<think>" in solution_str:
        # check if <tool> is used inside <think> tag
        inside_think_tag = solution_str.split("<think>")[-1].split("</think>")[0]
        if "<tool>" in inside_think_tag and "</tool>" in inside_think_tag:
            tool_reward = 1.

    # Bail out if the completion is invalid or incomplete.
    if incomplete:
        rwds = [-0.5, repetition_penalty_score, soft_format_reward, xml_reward, tool_reward] # -0.5 is a penalty for incomplete answers, at least an incomplete attempt is better than a wrong one I guess.
        rwd = sum([r*w for r, w in zip(rwds, weights)]) / sum(weights)
        return rwd, {"acc_reward_raw": 0., "acc_reward_scaled": -1., "repetition_penalty_score": repetition_penalty_score, "soft_format_reward": soft_format_reward, "xml_reward": xml_reward, "response_length": response_length, "progress": progress, "tool_use": tool_count}

    if invalid_think or no_think:
        tool_reward = 0. # If the model does not use the think tag, we don't want to reward it for using a tool.
        rwds = [-1., repetition_penalty_score, soft_format_reward, xml_reward, tool_reward]
        rwd = sum([r*w for r, w in zip(rwds, weights)]) / sum(weights)
        return rwd, {"acc_reward_raw": 0., "acc_reward_scaled": -1., "repetition_penalty_score": repetition_penalty_score, "soft_format_reward": soft_format_reward, "xml_reward": xml_reward, "response_length": response_length, "progress": progress, "tool_use": tool_count}

    if "</think>" not in solution_str: # If the completion does not contain the think tag, return -1 to encourage the model to use the think tag and prevent reward hacking.
        acc_reward = -1.
        acc_reward_raw = 0.
    else:
        min_value_wrong = -1.0
        max_value_wrong = -0.5
        # min_value_wrong = max_value_wrong = - 0.5
        min_value_correct = 0.5
        max_value_correct = 1.0

        # Remove the think tag from the completion only for training
        solution_str = solution_str.split("</think>")[-1]

        acc_reward_raw = compute_acc_reward(solution_str, ground_truth)

        # if acc_reward_raw>0.5:
        #     # If the completion is CORRECT, use the correct min/max values.
        #     min_value = min_value_correct
        #     max_value = max_value_correct
        #     progress = min(1, max(max_response_length - response_length, 0) / UNIT_LENGTH)
        # else:
        #     # Swap min/max for incorrect answers
        #     min_value = max_value_wrong
        #     max_value = min_value_wrong
        #     min_response_length = UNIT_LENGTH
        #     progress = min(1, max(max_response_length - response_length - min_response_length, 0) / (max_response_length - min_response_length))

        # # linear reward
        # acc_reward = min_value + progress * (max_value - min_value)

        # Apply cosine scaling based on length
        # progress = response_length / max_response_length

        if acc_reward_raw>0.5:
            min_value = min_value_correct
            max_value = max_value_correct
            progress = min(1, max(response_length - MAX_LENGTH_CONTROL, 0) / (max_response_length - MAX_LENGTH_CONTROL))
        else:
            # Swap min/max for incorrect answers
            min_value = max_value_wrong
            max_value = min_value_wrong
            progress = min(1, response_length / (max_response_length - UNIT_LENGTH))

        cosine = math.cos(progress * math.pi)
        acc_reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)


    metric = {
        "acc_reward_raw": acc_reward_raw,
        "acc_reward_scaled": acc_reward,
        "repetition_penalty_score": repetition_penalty_score,
        "soft_format_reward": soft_format_reward,
        "xml_reward": xml_reward,
        "response_length": response_length,
        "progress": progress,
        "tool_use": tool_count
    }

    return sum([r*w for r, w in zip([acc_reward, repetition_penalty_score, soft_format_reward, xml_reward, tool_reward], weights)]) / sum(weights), metric


# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    is_eq = str1.lower() == str2.lower() or grade_answer(str1, str2)
    if is_eq:
        return True

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)

        # normalize yes and true, no and false
        if ss1.lower() in {"yes", "true"}:
            ss1 = "__YES__"
        if ss1.lower() in {"no", "false"}:
            ss1 = "__NO__"
        
        if ss2.lower() in {"yes", "true"}:
            ss2 = "__YES__"
        if ss2.lower() in {"no", "false"}:
            ss2 = "__NO__"

        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return is_eq


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
