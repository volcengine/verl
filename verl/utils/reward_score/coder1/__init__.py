# SPDX-FileCopyrightText: (c) iSE UIUC Research Group
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Tuple, Optional
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from verl.utils.reward_score.coder1.firejail import code_exec_firejail as code_exec, _ERROR_MSG_PREFIX

_MAX_CHAR_DISPLAY = 2048


def exec_check_stdio(code, stdin, stdout, timeout=None):
    succ, output = code_exec(code=code, stdin=stdin, timeout=timeout)
    return succ, output, stdin, stdout


def validate_response_structure(processed_str: str) -> bool:
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    if match := think_pattern.match(processed_str):
        thinking = f"<think>{match.group(1)}</think>"
        response = (response.split(thinking)[-1].strip().strip("<answer>").strip("</answer>").strip())
        return ("<answer>" not in response and "</answer>" not in response and "<think>" not in response and
                "</think>" not in response)

    return False


def try_extract_answer(response: str) -> Tuple[Optional[str], str]:
    answer_pattern = r"<think>(.*?)</think>"
    if matches := list(re.finditer(answer_pattern, response, re.DOTALL)):
        thinking = f"<think>{matches[-1].group(1)}</think>"
        response = response.split(thinking)[-1].strip()
    return response.strip("<answer>").strip("</answer>").strip()


def extract_code_from_string(solution_str):
    pattern = re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL)
    code_blocks = pattern.findall(solution_str)
    return '\n'.join(code_blocks).strip()


def _check_fmt(response: str) -> Tuple[bool, str]:
    reward_log = ("-" * 16 + "Bad format detected!" + "-" * 16 + "\n" + "-" * 16 + "Original Model Output" + "-" * 16 +
                  "\n" + response)

    if not validate_response_structure(response):
        return False, reward_log

    if len(extract_code_from_string(try_extract_answer(response)).strip()) == 0:
        return False, reward_log

    return True, ""


def _check_correctness(solution_str, ground_truth, extra_info) -> Tuple[bool, str]:
    reward_log = []

    solution_code = extract_code_from_string(solution_str)

    reward_log.append("-" * 16 + "Extracted Code to Execute" + "-" * 16)
    reward_log.append(solution_code)

    t_start = time.time()

    ground_truth = json.loads(ground_truth)
    timeout = extra_info.get("timeout", None)
    if "pytest" in ground_truth or "functional" in ground_truth:
        if "functional" in ground_truth:
            succ, output = code_exec(solution_code + "\n" + ground_truth["functional"], timeout=timeout)
        else:  # pytest
            succ, output = code_exec(solution_code, pytest=ground_truth["pytest"], timeout=timeout)
        if not succ:
            reward_log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
            reward_log.append(output[:_MAX_CHAR_DISPLAY])
            reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
            reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
            return False, "\n".join(reward_log)
    elif "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list: str = ground_truth["inputs"]
        stdout_list: str = ground_truth["outputs"]

        # Add parallelism
        with ThreadPoolExecutor(max_workers=min(8, len(stdin_list))) as executor:
            futures = [
                executor.submit(exec_check_stdio, solution_code, stdin, stdout, timeout=timeout)
                for stdin, stdout in zip(stdin_list, stdout_list)
            ]
            for future in as_completed(futures):
                succ, output, stdin, stdout = future.result()
                if not succ or output.strip() != stdout.strip():
                    output = output[:_MAX_CHAR_DISPLAY]  # truncate output to print
                    reward_log.append("!" * 16 + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s" + "!" * 16)
                    reward_log.append(f"🔎Input: {repr(stdin)}")
                    reward_log.append(f"✅Expected: {repr(stdout.strip())}")
                    reward_log.append(
                        f"❌Actual: {output if output.startswith(_ERROR_MSG_PREFIX) else repr(output.strip())}")
                    reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
                    reward_log.append(extra_info["prompt"].replace("\n\n", "\n"))
                    return False, "\n".join(reward_log)
    else:
        raise ValueError(
            f"Current supports for ground-truth are ['pytest', 'functional', 'inputs/outputs'] -- No idea what's: {ground_truth = }"
        )

    reward_log.append("+" * 16 + "Test Execution Passed! (Output)" + "+" * 16)
    reward_log.append(output)
    return True, "\n".join(reward_log)


def _compute_score(solution_str: str, ground_truth: str, extra_info: dict):
    """
    Compute the score for the solution string based on the ground truth and extra information.
    Args:
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): A JSON string representing the ground truth.
        extra_info (dict): Additional information for evaluation.
        format_reward (float): Reward for correct formatting.
        answer_reward (float): Reward for correct answer.
    Returns:
        float: The computed score.
        str: The log of the reward calculation process.
    """
    reward_log = []
    reward_log.append("-" * 16 + "Original Model Output" + "-" * 16)
    reward_log.append(solution_str)

    # Check format
    fmt_succ, fmt_log = _check_fmt(solution_str)
    if not fmt_succ:
        reward_log.append(fmt_log)
        return 0.0, "\n".join(reward_log)

    # Check correctness
    correct_succ, correct_log = _check_correctness(solution_str, ground_truth, extra_info)
    reward_log.append(correct_log)
    if not correct_succ:
        return 0.1, "\n".join(reward_log)

    return 1.0, "\n".join(reward_log)


def compute_score(solution_str, ground_truth, extra_info):
    if isinstance(extra_info, np.ndarray):
        extra_info = extra_info.item()
    score, reward_log = _compute_score(solution_str, ground_truth, extra_info=extra_info)
    marker = "✅" if score == 1 else "❌"
    reward_log = marker * 16 + "Reward Calculation" + marker * 16 + "\n" + reward_log + "\n" + marker * 16 + f"Final Rward = {score}" + marker * 16
    print(reward_log + "\n\n")
    return score
