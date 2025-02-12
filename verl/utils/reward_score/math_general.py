import aiohttp
import asyncio
import time
import re
from .grader import grade_answer

url = "https://verifier.yuewu.ml/api"
headers = {
    "Content-Type": "application/json",
    "Authentication": "RCbUvAw8nEv_jAuQa82uvAoZBiUv0fMEc28FUddmh78"  # Replace with your API key if needed.
}

async def compute_acc_reward(solution_str, ground_truth):
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
                # print("Error in compute_acc_reward:")
                # print(string_in_last_boxed)
                # print(e)
                # return 0.

    if extracted_sol is None:
        return 0.

    payload = {
        "answer": extracted_sol,
        "ground_truth": ground_truth
    }
    delay = 0.1   # initial delay (in seconds)
    max_delay = 5 # maximum delay (in seconds)
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.post(url, headers=headers, json=payload, timeout=600) as response:
                    resp_json = await response.json()
                    retval = resp_json['LaTeXAgreementScore']
                    return retval
            except aiohttp.ClientError as e:
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

def compute_format_reward(solution_str):
    """Returns 0.5 if the completion is in the correct format, 0. if not."""

    # Use re to check if the completion contains \boxed{...}
    if re.search(r"\\boxed{.*?}", solution_str):
        return 0.5
    
    return 0.

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\s*$"
    responses = [completion for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [1. if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
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
    pattern = r"<answer>.*?</answer>"
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 1:
        reward += 0.5

        reward -= (len(text.split("</answer>")[-1]) - 1)*0.001

    return reward

async def compute_score(solution_str, ground_truth):
    """Reward function that checks if the completion is the same as the ground truth."""

    split, ground_truth = ground_truth.split("######")

    # Remove the prompt from the completion.
    if "<|im_start|>assistant<|im_end|>" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant<|im_end|>")[1]
    solution_str = solution_str.replace("<|im_end|>", "").strip()


    # If the split is test, we can directly compare the completion with the ground truth.
    if split == "test":
        return await compute_acc_reward(solution_str, ground_truth)

    strict_format_reward = strict_format_reward_func([solution_str])[0]
    soft_format_reward = soft_format_reward_func([solution_str])[0]
    # xml_reward = count_xml(solution_str)
    xml_reward = strict_xml(solution_str)
    acc_reward = await compute_acc_reward(solution_str, ground_truth)


    weights = [2, 0.5, 0.5, 0.25]

    return sum([r*w for r, w in zip([acc_reward, strict_format_reward, soft_format_reward, xml_reward], weights)]) / sum(weights)


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

    is_eq = str1 == str2 or grade_answer(str1, str2)
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
