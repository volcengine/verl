# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import multiprocessing
import warnings
from functools import partial

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass
    return ret_score


def _evaluation_worker(model_output, ground_truth, evaluation_func):
    try:
        return evaluation_func(model_output, ground_truth)
    except Exception as e:
        warnings.warn(f"An unexpected error occurred in worker: {e}")
        return 0.0


def parallel_compute_score(model_outputs: list, ground_truths: list, timeout: int = 20, num_processes: int = 32) -> list:
    tasks_to_process = list(enumerate(zip(model_outputs, ground_truths)))
    results = [-1.0] * len(model_outputs)
    worker_func = partial(_evaluation_worker, evaluation_func=compute_score)

    while any(r == -1.0 for r in results):
        remaining_tasks = [(i, args) for i, args in tasks_to_process if results[i] == -1.0]
        
        pool = None 
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                async_results = {
                    original_index: pool.apply_async(worker_func, args=task_args)
                    for original_index, task_args in remaining_tasks
                }

                for original_index, res in async_results.items():
                    try:
                        score = res.get(timeout=timeout)
                        results[original_index] = score
                    except multiprocessing.TimeoutError:
                        warnings.warn(f"Timeout when processing item at index {original_index}. Will restart pool.")
                        raise multiprocessing.TimeoutError 
                    except Exception as e:
                        warnings.warn(f"An error occurred retrieving result at index {original_index}: {e}")
                        results[original_index] = 0.0 

        except multiprocessing.TimeoutError:
            # This block is executed after the inner loop raises TimeoutError.
            # Upon exiting the 'with' block, `pool.terminate()` is called automatically,
            # which forcibly kills all subprocesses to prevent zombies.
            print("Pool timed out. A new pool will be created for remaining tasks.")
            # Continue the outer while loop to process the remaining tasks with a new pool.
            continue 

    return results
