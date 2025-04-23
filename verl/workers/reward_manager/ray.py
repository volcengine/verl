import os
import time
import logging
import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import importlib
import ray
from ray.util.actor_pool import ActorPool
import torch
from verl import DataProto

# default reward fn import
from verl.utils.reward_score import _default_compute_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Remote Actor for Scoring
# ----------------------------
@ray.remote(num_cpus=1)
class ScoreActor:
    def __init__(self, module_path: str, fn_name: str):
        print("ScoreActor initializing with %s.%s", module_path, fn_name)
        module = importlib.import_module(module_path)
        self._score_fn: Callable[..., Union[float, Dict[str, Any]]] = getattr(module, fn_name)

    def score_slice(
        self,
        slice_: List[Tuple[str, str, Any, Optional[Dict[str, Any]]]],
    ) -> List[Union[float, Dict[str, Any]]]:
        start = time.time()
        print("ScoreActor received slice of size %d", len(slice_))
        results = [
            self._score_fn(
                data_source=ds,
                solution_str=sol,
                ground_truth=gt,
                extra_info=ei,
            )
            for ds, sol, gt, ei in slice_
        ]
        elapsed = time.time() - start
        print("ScoreActor finished scoring slice in %.2fs", elapsed)
        return results

# ----------------------------
# Parallel Reward Manager
# ----------------------------
class RayRewardManager:
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        *,
        module_path: Optional[str] = None,
        fn_name: Optional[str] = None,
        reward_fn_key: str = "data_source",
        task_timeout: float = 60.0,
        pool_size: Optional[int] = None,
        **kwargs
    ) -> None:
        # Warn and ignore extra kwargs
        if kwargs:
            print(
                "RayRewardManager received unexpected kwargs and will ignore them: %s",
                kwargs,
            )

        # Initialize Ray with runtime_env to ship code
        print("Initializing Ray with working_dir %s", os.getcwd())
        ray.init(
            logging_level=logging.ERROR,
            ignore_reinit_error=True,
            runtime_env={"working_dir": os.getcwd()},
        )

        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.task_timeout = task_timeout

        # Determine which scoring function to use
        if compute_score is not None:
            target_fn = compute_score
            print("Using provided compute_score function %s.%s", target_fn.__module__, target_fn.__name__)
        elif module_path and fn_name:
            module = importlib.import_module(module_path)
            target_fn = getattr(module, fn_name)
            print("Using compute_score from module %s.%s", module_path, fn_name)
        else:
            target_fn = _default_compute_score
            print("Using default compute_score _default_compute_score")

        # Derive module path and function name for actor import
        self.module_path = target_fn.__module__
        self.fn_name = target_fn.__name__

        # Determine pool size based on available CPUs
        total_cpus = int(ray.cluster_resources().get("CPU", 1))
        default_pool = max(1, min(256, total_cpus-8))
        self.pool_size = pool_size or default_pool
        print("Actor pool size set to %d", self.pool_size)

        # Build actor pool
        self.actors = [ScoreActor.remote(self.module_path, self.fn_name) for _ in range(self.pool_size)]
        self.pool = ActorPool(self.actors)
        print("Initialized ActorPool with %d actors", len(self.actors))

        # Metrics
        self.metrics: Dict[str, Any] = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "timeout_tasks": 0,
            "total_time": 0.0,
            "avg_task_time": 0.0,
        }

    def __del__(self):
        for actor in getattr(self, "actors", []):
            try:
                ray.kill(actor, no_restart=True)
                print("Killed actor %s", actor)
            except Exception as e:
                print("Failed to kill actor: %s", e)

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        # If precomputed scores present, return them
        if "rm_scores" in data.batch:
            scores = data.batch["rm_scores"]
            print("Returning precomputed rm_scores of shape %s", tuple(scores.shape))
            return {"reward_tensor": scores} if return_dict else scores

        # Prepare containers
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra: Dict[str, List[Any]] = defaultdict(list)

        # Build task inputs
        task_inputs, aux_info = self._prepare_task_inputs(data)
        num_tasks = len(task_inputs)
        print("Prepared %d tasks for scoring", num_tasks)
        if num_tasks == 0:
            return {"reward_tensor": reward_tensor} if return_dict else reward_tensor

        # Create slices for actors
        slice_size = max(1, math.ceil(num_tasks / self.pool_size))
        slices = [task_inputs[i : i + slice_size] for i in range(0, num_tasks, slice_size)]
        aux_slices = [aux_info[i : i + slice_size] for i in range(0, num_tasks, slice_size)]
        print("Divided tasks into %d slices of up to %d tasks each", len(slices), slice_size)

        # Scatter & gather with timeout handling
        print("Dispatching slices to actors with timeout=%.2fs", self.task_timeout)
        results_iter = self.pool.map(
            lambda actor, batch: actor.score_slice.remote(batch),
            slices,
            timeout_seconds=self.task_timeout,
        )
        flat_results, flat_aux = [], []
        for idx_slice, info_batch in enumerate(aux_slices):
            try:
                result_batch = next(results_iter)
                print("Received result batch %d of size %d", idx_slice, len(result_batch))
            except StopIteration:
                print("ActorPool iterator exhausted at slice %d", idx_slice)
                break
            except (ray.exceptions.GetTimeoutError,) as e:
                print(
                    "Slice %d timed out after %.2fs: %s", idx_slice, self.task_timeout, e
                )
                self.metrics["timeout_tasks"] += len(info_batch)
                result_batch = [None] * len(info_batch)
            flat_results.extend(result_batch)
            flat_aux.extend(info_batch)

        # Process results
        print("Processing %d total results", len(flat_results))
        self._process_results(flat_results, flat_aux, reward_tensor, reward_extra)
        print(
            "Scoring complete: total_tasks=%d completed=%d failed=%d timed_out=%d avg_task_time=%.2fs",
            self.metrics["total_tasks"], self.metrics["completed_tasks"],
            self.metrics["failed_tasks"], self.metrics["timeout_tasks"], self.metrics["avg_task_time"],
        )
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra)}
        return reward_tensor

    def _prepare_task_inputs(
        self, data: DataProto
    ) -> Tuple[List[Tuple[Any, ...]], List[Dict[str, Any]]]:
        task_inputs: List[Tuple[Any, ...]] = []
        aux_info: List[Dict[str, Any]] = []
        for i, item in enumerate(data):
            # Decode prompt & response
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = int(item.batch["attention_mask"][:prompt_len].sum())
            prompt_str = self.tokenizer.decode(
                prompt_ids[-valid_prompt_len:], skip_special_tokens=True
            )
            resp_ids = item.batch["responses"]
            valid_resp_len = int(item.batch["attention_mask"][prompt_len:].sum())
            resp_str = self.tokenizer.decode(
                resp_ids[:valid_resp_len], skip_special_tokens=True
            )
            if resp_str.endswith(self.tokenizer.eos_token):
                resp_str = resp_str[: -len(self.tokenizer.eos_token)]

            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch[self.reward_fn_key]
            extra_info = item.non_tensor_batch.get("extra_info")

            task_inputs.append((data_source, resp_str, ground_truth, extra_info))
            aux_info.append({
                "index": i,
                "valid_response_length": valid_resp_len,
                "data_source": data_source,
                "prompt_str": prompt_str,
                "response_str": resp_str,
                "ground_truth": ground_truth,
            })
        print("_prepare_task_inputs created %d inputs", len(task_inputs))
        return task_inputs, aux_info

    def _process_results(
        self,
        results: List[Optional[Union[float, Dict[str, Any]]]],
        aux_info: List[Dict[str, Any]],
        reward_tensor: torch.Tensor,
        reward_extra: Dict[str, List[Any]],
    ) -> None:
        start = time.time()
        print("Starting _process_results for %d items", len(results))
        for res, info in zip(results, aux_info):
            self.metrics["total_tasks"] += 1
            idx = info["index"]
            vr_len = info["valid_response_length"]
            if res is None:
                self.metrics["failed_tasks"] += 1
                print("Task %d failed or timed out for source %s", idx, info.get("data_source"))
                score = 0.0
            else:
                self.metrics["completed_tasks"] += 1
                if isinstance(res, dict):
                    score = float(res.get("score", 0.0))
                    for k, v in res.items():
                        reward_extra[k].append(v)
                else:
                    score = float(res)
            if vr_len > 0:
                reward_tensor[idx, vr_len - 1] = score
            if getattr(self, "_printed", 0) < self.num_examine:
                self._print_debug_info(info, res, score)
        elapsed = time.time() - start
        self.metrics["total_time"] += elapsed
        if self.metrics["total_tasks"]:
            self.metrics["avg_task_time"] = self.metrics["total_time"] / self.metrics["total_tasks"]
        print(
            "_process_results complete: total=%d completed=%d failed=%d timed_out=%d avg_time=%.2fs",
            self.metrics["total_tasks"], self.metrics["completed_tasks"],
            self.metrics["failed_tasks"], self.metrics["timeout_tasks"], self.metrics["avg_task_time"],
        )

    def _print_debug_info(self, info: Dict[str, Any], result: Any, reward: float) -> None:
        self._printed = getattr(self, "_printed", 0) + 1
        print("[prompt] %s", info["prompt_str"])
        print("[response] %s", info["response_str"])
        print("[ground_truth] %s", info["ground_truth"])
        if isinstance(result, dict):
            for k, v in result.items():
                print("[%s] %s", k, v)
        else:
            print("[score] %s", result)
        print("[final_reward] %.3f", reward)
