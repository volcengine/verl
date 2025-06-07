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

import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import psutil
import torch

from verl import DataProto

from verl.workers.reward_manager.base import BaseRewardManager


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.0):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        future = loop.run_in_executor(
            executor,
            partial(evaluation_func, task, completion, reference, task_extra_info)
        )
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"[Error] Task failed: {e}, completion: {completion[:80]}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    if extra_info is None:
        extra_info = [None] * len(tasks)
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            # Create tasks for all rows
            tasks_async = [
                single_compute_score(evaluation_func, c, r, t, ei, executor, timeout=300.0)
                for c, r, t, ei in zip(completions, references, tasks, extra_info)
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result, (int, float, bool)):
            scores.append(float(result))
        else:
            scores.append(float(result[0]))
    return scores

def run_reward_scoring(evaluation_func, completions, references, tasks, extra_info=None, num_processes=64):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(parallel_compute_score_async(
            evaluation_func, completions, references, tasks, extra_info, num_processes
        ))
    finally:
        loop.close()


class PrimeRewardManager(BaseRewardManager):
    def compute_scores(self, reward_data: DataProto) -> list[int | float | dict]:
        try:
            scores = run_reward_scoring(
                self.user_defined_compute_scores,
                completions=reward_data.non_tensor_batch["solution_strs"],
                references=reward_data.non_tensor_batch["ground_truths"],
                tasks=reward_data.non_tensor_batch["data_sources"],
                extra_info=reward_data.non_tensor_batch["extra_infos"],
                num_processes=64,
            )
        except asyncio.TimeoutError:
            print("[Timeout] Global reward scoring timed out. Setting all as 0.")
            scores = [0.0 for _ in range(len(reward_data))]
        except Exception as e:
            print(f"[Error] Unexpected error during scoring. Setting all as 0. {e}")
            scores = [0.0 for _ in range(len(reward_data))]
        return scores

    def postprocess_scores(self, reward_data: DataProto, training_data: DataProto) -> None:
        training_data.batch["acc"] = torch.tensor(reward_data.batch["scores"], device=training_data.batch["prompts"].device)
        super().postprocess_scores(reward_data, training_data)
