import os
import time
import logging
import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import importlib
import ray
import torch
from verl import DataProto
from verl.utils.reward_score import _default_compute_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@ray.remote(max_calls=256) # keep warm
def _score_slice_task(
    module_path: str,
    fn_name: str,
    slice_: List[Tuple[str, str, Any, Optional[Dict[str, Any]]]]
) -> List[Union[float, Dict[str, Any]]]:
    """
    Remote function to score a batch (slice) of inputs using the given module path and fn name.
    """
    module = importlib.import_module(module_path)
    fn: Callable[..., Union[float, Dict[str, Any]]] = getattr(module, fn_name)
    return [
        fn(data_source=ds, solution_str=sol, ground_truth=gt, extra_info=ei)
        for ds, sol, gt, ei in slice_
    ]

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
        task_timeout: float = 300.0,
        pool_size: Optional[int] = None,
        **kwargs
    ) -> None:
        if kwargs:
            logger.warning("Ignoring unexpected kwargs: %s", kwargs)
            
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.task_timeout = task_timeout

        # Select scoring function
        if compute_score:
            fn = compute_score
        elif module_path and fn_name:
            module = importlib.import_module(module_path)
            fn = getattr(module, fn_name)
        else:
            fn = _default_compute_score

        self.module_path = fn.__module__
        self.fn_name = fn.__name__

        # Determine parallelism
        total_cpus = int(ray.cluster_resources().get("CPU", 1))
        self.pool_size = pool_size or max(1, min(256, total_cpus))
        logger.info(
            "RayRewardManager using %s.%s across %d parallel tasks",
            self.module_path,
            self.fn_name,
            self.pool_size,
        )

        # Metrics
        self.metrics: Dict[str, Union[int, float]] = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "timeout_tasks": 0,
            "total_time": 0.0,
            "avg_task_time": 0.0,
        }

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        # Use precomputed scores if available
        if "rm_scores" in data.batch:
            scores = data.batch["rm_scores"]
            return ({"reward_tensor": scores} if return_dict else scores)

        # Prepare output tensors
        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )
        reward_extra: Dict[str, List[Any]] = defaultdict(list)

        # Prepare inputs and auxiliary info
        inputs, aux = self._prepare_task_inputs(data)
        n = len(inputs)
        if n == 0:
            return ({"reward_tensor": reward_tensor} if return_dict else reward_tensor)

        # Split into slices for parallel tasks
        sz = max(1, math.ceil(n / self.pool_size))
        slices = [inputs[i : i + sz] for i in range(0, n, sz)]
        aux_slices = [aux[i : i + sz] for i in range(0, n, sz)]

        # Dispatch remote tasks
        tasks = [
            _score_slice_task.remote(self.module_path, self.fn_name, s)
            for s in slices
        ]
        ref_to_aux = dict(zip(tasks, aux_slices))

        # Wait for all or timeout
        done, pending = ray.wait(
            tasks, timeout=self.task_timeout, num_returns=len(tasks)
        )

        # Cancel and count timed-out tasks
        if pending:
            for t in pending:
                ray.cancel(t)
                self.metrics["timeout_tasks"] += len(ref_to_aux[t])

        # Gather results
        flat_res: List[Any] = []
        flat_aux: List[Dict[str, Any]] = []
        for t in done:
            try:
                out = ray.get(t)
                self.metrics["completed_tasks"] += len(out)
            except Exception:
                aux_batch = ref_to_aux[t]
                out = [None] * len(aux_batch)
                self.metrics["failed_tasks"] += len(out)

            flat_res.extend(out)
            flat_aux.extend(ref_to_aux[t])

        # Process all results
        self._process_results(flat_res, flat_aux, reward_tensor, reward_extra)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": dict(reward_extra)}
        return reward_tensor

    def _prepare_task_inputs(
        self, data: DataProto
    ) -> Tuple[
        List[Tuple[str, str, Any, Optional[Dict[str, Any]]]],
        List[Dict[str, Any]]
    ]:
        inputs: List[Tuple[str, str, Any, Optional[Dict[str, Any]]]] = []
        aux: List[Dict[str, Any]] = []
        for i, item in enumerate(data):
            pids = item.batch["prompts"]
            plen = pids.shape[-1]
            vplen = int(item.batch["attention_mask"][:plen].sum())
            pstr = self.tokenizer.decode(
                pids[-vplen:], skip_special_tokens=True
            )

            rids = item.batch["responses"]
            vrl = int(item.batch["attention_mask"][plen:].sum())
            rstr = self.tokenizer.decode(rids[:vrl], skip_special_tokens=True)
            if rstr.endswith(self.tokenizer.eos_token):
                rstr = rstr[: -len(self.tokenizer.eos_token)]

            gt = item.non_tensor_batch["reward_model"]["ground_truth"]
            ds = item.non_tensor_batch[self.reward_fn_key]
            ei = item.non_tensor_batch.get("extra_info")

            inputs.append((ds, rstr, gt, ei))
            aux.append({
                "index": i,
                "valid_response_length": vrl,
                "data_source": ds,
                "prompt_str": pstr,
                "response_str": rstr,
                "ground_truth": gt,
            })

        return inputs, aux

    def _process_results(
        self,
        results: List[Union[float, Dict[str, Any]]],
        aux: List[Dict[str, Any]],
        reward_tensor: torch.Tensor,
        reward_extra: Dict[str, List[Any]],
    ) -> None:
        start = time.time()
        # reset print counter
        self._printed = 0
        for res, info in zip(results, aux):
            self.metrics["total_tasks"] += 1
            idx = info["index"]
            vrl = info["valid_response_length"]

            if res is None:
                score = 0.0
            elif isinstance(res, dict):
                score = float(res.get("score", 0.0))
                for k, v in res.items():
                    reward_extra[k].append(v)
            else:
                score = float(res)

            if vrl > 0:
                reward_tensor[idx, vrl - 1] = score

            if getattr(self, "_printed", 0) < self.num_examine:
                self._print_debug_info(info, res, score)

        elapsed = time.time() - start
        self.metrics["total_time"] += elapsed
        if self.metrics["total_tasks"]:
            self.metrics["avg_task_time"] = (
                self.metrics["total_time"] / self.metrics["total_tasks"]
            )

    def _print_debug_info(
        self,
        info: Dict[str, Any],
        result: Union[float, Dict[str, Any]],
        reward: float,
    ) -> None:
        self._printed = getattr(self, "_printed", 0) + 1
        logger.info("[prompt] %s", info["prompt_str"])
        logger.info("[response] %s", info["response_str"])
        logger.info("[ground_truth] %s", info["ground_truth"])

        # log extra info using safe .get
        extra_info = info.get("extra_info", {})
        if extra_info:
            logger.info("[extra_info] %s", extra_info)

        if isinstance(result, dict):
            for k, v in result.items():
                logger.info("[%s] %s", k, v)
        else:
            logger.info("[score] %s", result)
        logger.info("[final_reward] %.3f", reward)
