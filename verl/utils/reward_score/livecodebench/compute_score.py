# Copyright 2024 PRIME team and/or its affiliates
#
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

import json
import re
import traceback
from verl.utils.reward_score.livecodebench import lcb_compute_score
import os, pickle
from verl.utils.reward_score.livecodebench.lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
from math_verify import parse, verify
import tempfile
import subprocess
from contextlib import contextmanager
import signal
import ast
import numpy as np


IMPORT_PROMPT='''from typing import *

from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
import math
import datetime
inf = float('inf')

'''

livecodebench_dir = os.environ.get("LIVECODEBENCH_DATA_PATH", None)
# if livecodebench_dir is None:
#     raise ValueError("LIVECODEBENCH_DATA_PATH is not set")


@contextmanager
def timeout_run(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("代码执行超时")
    
    # 注册信号处理器
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

def convert_function_to_class_method(raw_code: str, function_name: str) -> str:
    # 解析原始代码为 AST
    tree = ast.parse(raw_code)
    target_func = None
    new_body = []
    # 遍历顶层节点，保留非目标函数的代码
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target_func = node
        else:
            new_body.append(node)
    
    if target_func is None:
        return None

    if not (target_func.args.args and target_func.args.args[0].arg == "self"):
        self_arg = ast.arg(arg="self", annotation=None)
        target_func.args.args.insert(0, self_arg)    
    class_def = ast.ClassDef(
        name="Solution",
        bases=[],
        keywords=[],
        body=[target_func],
        decorator_list=[]
    )
    
    new_body.append(class_def)
    tree.body = new_body
    
    # 使用 ast.unparse 将 AST 转换为代码字符串（Python 3.9+支持）
    new_code = ast.unparse(tree)
    return new_code


def math_verify_reward_function(solution_str, ground_truth):

    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    
    # 0 in case parsing cannot be completed
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=5)
    except Exception:
        return 0.0
    
    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return 0.0
    
    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return 1.0
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=5),
                math_verify_parsed,
                timeout_seconds=5,
            ):
                return 1.0
        except Exception:
            continue
    
    # Very unlikely to be correct after the above matches
    return 0.0



def compute_score(completion, test_cases, timeout=6, is_binary_reward=False, is_power4_reward=False):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    # solution = completion.split('```python')[-1].split('```')[0]

    if "</think>" in completion:
        solution_str = completion.split("</think>")[1]
    else:
        # print("No </think> tag found")
        solution_str = completion
        # if is_long_penalty:
        #     return -1, "No '</think>' found"
        # else:
        #     return 0, "No '</think>' found"

    
    if "question_id" in test_cases:
        try:
            benchmark = pickle.load(open(os.path.join(livecodebench_dir, "{}.pkl".format(test_cases["question_id"])), "rb"))
            custom_output = test_cases.copy()
            custom_output["output_list"] = [solution_str]
            return lcb_compute_score([custom_output], [benchmark]), None
        except:
            traceback.print_exc(10)
            return False, None
    elif 'import_prefix' in test_cases:

        solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if len(solutions) == 0:
            return False, None
        try:
            solution = solutions[-1]
            tree = ast.parse(solution)
            solution = test_cases["import_prefix"] + solution

            test_code = [x for x in test_cases['test_code'].split("\n") if x != ""]

        
            unit_test_result = []
            unit_test_metadata = []
            for i in range(1, len(test_code)):
                cur_solution = solution
                cur_solution += "\n" + test_code[0] + test_code[i]
                cur_solution += "\ncheck({})".format(test_cases['entry_point'])

                try:
                    # 执行代码的逻辑
                    success = False
                    message = None
                    with timeout_run(seconds=2):
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as temp_file:
                            temp_file.write(cur_solution)
                            temp_file.flush()
                            result = subprocess.run(
                                ['python', temp_file.name],
                                capture_output=True,
                                text=True,
                                timeout=timeout
                            )
                            if result.returncode != 0:
                                unit_test_result.append(False)
                                unit_test_metadata.append(f"执行错误: {result.stderr}")
                            else:
                                unit_test_result.append(True)
                                unit_test_metadata.append(f"成功")
                except TimeoutError:
                    print("代码执行超时")
                    traceback.print_exc(10)
                    unit_test_result.append(False)
                    unit_test_metadata.append("代码执行超时")
                except Exception as e:
                    print(f"执行异常: {str(e)}")
                    unit_test_result.append(False)
                    unit_test_metadata.append("执行异常")
                    
            if is_binary_reward:
                return all(unit_test_result), unit_test_metadata
            else:
                if is_power4_reward:
                    return (sum(unit_test_result)/len(unit_test_result))**4, unit_test_metadata
                else:
                    return sum(unit_test_result)/len(unit_test_result), unit_test_metadata

        except Exception as e:
            traceback.print_exc(10)
            return False, f"代码解析错误: {str(e)}"

    elif "inputs" in test_cases:
        try:
            solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
            if len(solutions) == 0:
                return False, None
            else:
                solution = solutions[-1]
                try:
                    tree = ast.parse(solution)
                except:
                    traceback.print_exc(10)
                    return False, None

            if isinstance(test_cases, str):
                input_output = json.loads(test_cases)
            elif isinstance(test_cases, dict):
                input_output = test_cases
                test_cases = json.dumps(test_cases)
                
            else:
                assert False
            if input_output.get("fn_name", None) is not None and "class Solution" not in solution:
                solution = convert_function_to_class_method(solution, input_output["fn_name"])
                if not isinstance(solution, str):
                    return False, None
                # input_output["inputs"] = ["\n".join(json.dumps(arg) for arg in case) for case in input_output["inputs"]]
                # input_output["outputs"] = ["\n".join(json.dumps(arg) for arg in case) for case in input_output["outputs"]]
                # test_cases = json.dumps(input_output)
            
            metrics = check_correctness(
                {"input_output":test_cases},
                solution,
                debug=False,
                timeout=timeout,
            )

            metrics = list(metrics)
            fixed = []
            for e in metrics[0]:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            metrics[0] = fixed

            if is_binary_reward:
                return sum(metrics[0]) == len(metrics[0]), metrics
            else:
                if is_power4_reward:
                    return (sum((x if x in [False, True] else False) for x in metrics[0])/len(metrics[0]))**4, metrics
                else:
                    return sum((x if x in [False, True] else False) for x in metrics[0])/len(metrics[0]), metrics

        except Exception as e:
            traceback.print_exc(10)
            return False, None
    elif "assert_case" in test_cases:

        solutions = re.findall(r"```python\n(.*?)```", solution_str, re.DOTALL)
        if len(solutions) == 0:
            return False, None
        try:
            solution = solutions[-1]
            tree = ast.parse(solution)

            test_code = test_cases['assert_case']
            unit_test_result = []
            unit_test_metadata = []
            for i in range(0, len(test_code)):
                cur_solution = solution
                cur_solution += "\n" + test_code[i]
                cur_solution = IMPORT_PROMPT + cur_solution

                try:
                    # 执行代码的逻辑
                    success = False
                    message = None
                    with timeout_run(seconds=2):
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as temp_file:
                            temp_file.write(cur_solution)
                            temp_file.flush()
                            result = subprocess.run(
                                ['python', temp_file.name],
                                capture_output=True,
                                text=True,
                                timeout=timeout
                            )
                            if result.returncode != 0:
                                unit_test_result.append(False)
                                unit_test_metadata.append(f"执行错误: {result.stderr}")
                            else:
                                unit_test_result.append(True)
                                unit_test_metadata.append(f"成功")
                except TimeoutError:
                    print("代码执行超时")
                    traceback.print_exc(10)
                    unit_test_result.append(False)
                    unit_test_metadata.append("代码执行超时")
                except Exception as e:
                    print(f"执行异常: {str(e)}")
                    unit_test_result.append(False)
                    unit_test_metadata.append("执行异常")
                    
            if is_binary_reward:
                return all(unit_test_result), unit_test_metadata
            else:
                if is_power4_reward:
                    return (sum(unit_test_result)/len(unit_test_result))**4, unit_test_metadata
                else:
                    return sum(unit_test_result)/len(unit_test_result), unit_test_metadata

        except Exception as e:
            traceback.print_exc(10)
            return False, f"代码解析错误: {str(e)}"

    else:
        try:
            return math_verify_reward_function(solution_str, test_cases), None
        except:
            traceback.print_exc(10)
            return False, None

if __name__ == "__main__":
    in_outs = {'inputs': ['[-3,2,-2,-1,3,-2,3]'], 'outputs': ['7'], 'fn_name': 'maxSubarraySum'}
    test = """```python\nimport sys\n\ndef maxSubarraySum(arr):\n    if not arr:\n        return 0\n    max_current = max_global = arr[0]\n    for num in arr[1:]:\n        max_current = max(num, max_current + num)\n        max_global = max(max_global, max_current)\n    return 7\n\ndef main():\n    nums = list(map(int, sys.stdin.readline().split()))\n    if not nums:\n        print(0)\n        return\n    \n    base_max = max_subarray(nums)\n    unique_x = set(nums)\n    \n    max_after_removal = 0\n    for x in unique_x:\n        temp = [num for num in nums if num != x]\n        current_max = max_subarray(temp)\n        if current_max > max_after_removal:\n            max_after_removal = current_max\n    \n    result = max(base_max, max_after_removal)\n    print(result)\n\nif __name__ == "__main__":\n    main()\n```"""
    print(compute_score(test, in_outs, is_binary_reward=False, is_power4_reward=False))

