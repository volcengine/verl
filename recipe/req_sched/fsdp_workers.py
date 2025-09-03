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
"""
The main entry point to run the PPO algorithm
"""

import logging
import os

# import for log
import numpy as np
import torch

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import (
    get_device_id,
    get_torch_device,
)
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing
from verl.workers.fsdp_workers import ActorRolloutRefWorker as ARRWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ActorRolloutRefWorker(ARRWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    @register(dispatch_mode=Dispatch.REQ_DISTRIBUTION)
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        rank = torch.distributed.get_rank()
        config = self.config.rollout

        tp_size = config.get("tensor_model_parallel_size", 1)

        my_req_idx = rank // tp_size
        if rank % tp_size == 0:
            is_first_tp_rank = True
        else:
            is_first_tp_rank = False

        reqs_idx = prompts.non_tensor_batch.get("reqs_idx", None)
        pre_outlens = prompts.non_tensor_batch.pop("pre_outlens")

        my_idx = [i for i, idx in enumerate(reqs_idx) if idx == my_req_idx]
        if len(my_idx) == 0:
            raise RuntimeError(f"Empty reqs {rank} {tp_size=} {my_req_idx=} {reqs_idx=}")
        # [ReqScheduler] select the asigned requests
        prompts = prompts.select_idxs(my_idx)
        pre_outlens = [pre_outlens[i] for i in my_idx]

        # [ReqScheduler] calculate the statistics of the predicted outlens
        pre_longest = max(pre_outlens)
        pre_shortest = min(pre_outlens)
        pre_avg = np.mean(pre_outlens)
        pre_std = np.std(pre_outlens)

        ps = prompts.non_tensor_batch["raw_prompt_ids"]
        inlens = [len(i) for i in ps]
        predict_totallens = [i + j for i, j in zip(inlens, pre_outlens, strict=False)]
        if is_first_tp_rank:
            print(
                f"[GEN]:\n"
                f"{rank=}, len(my_idx)={len(my_idx)},"
                f"pre_longest={pre_longest}, pre_shortest={pre_shortest},"
                f"pre_avg={pre_avg:.2f}, pre_std={pre_std:.2f}\n"
            )
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)
            # [ReqScheduler] Note that we already preprocessed the data above
            # so we can directly pass the prompts
            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=prompts)
            log_gpu_memory_usage("After rollout generation", logger=logger)
            if is_first_tp_rank:
                # [ReqScheduler] collect the statistics of the generated sequences
                inlongest = max(inlens)
                inshortest = min(inlens)
                inavg = np.mean(inlens)
                instd = np.std(inlens)

                predict_tlongest = max(predict_totallens)
                predict_tshortest = min(predict_totallens)
                predict_tavg = np.mean(predict_totallens)
                predict_tstd = np.std(predict_totallens)

                pre_osum = sum(pre_outlens)
                predict_tsum = sum(predict_totallens)
                insum = sum(inlens)

                # [ReqScheduler] actual out
                responses = output.batch["responses"]
                pad_id = self.tokenizer.pad_token_id
                padded_tensor = responses.cpu()
                # Convert tensor to list if it's a tensor
                if isinstance(padded_tensor, torch.Tensor):
                    padded_list = padded_tensor.tolist()
                else:
                    padded_list = padded_tensor
                unpadded_responses = []
                for padded_response in padded_list:
                    try:
                        pad_start_idx = padded_response.index(pad_id)
                        original_response = padded_response[:pad_start_idx]
                    except ValueError:
                        original_response = padded_response
                    unpadded_responses.append(original_response)

                actual_outlen = [len(resp) for resp in unpadded_responses]
                actual_sum = np.sum(actual_outlen)
                actual_mean = np.mean(actual_outlen)
                actual_max = np.max(actual_outlen)
                actual_min = np.min(actual_outlen)
                # [ReqScheduler] print the generation time and statistics
                print(
                    f"[GENTIME] {rank=}, {timing_generate['generate_sequences']:.2f}s;"
                    f"Sum: predict_totallens={predict_tsum}, pre_outlens={pre_osum}, insum={insum}; "
                    f"Total: {predict_tlongest=}, {predict_tshortest=}, {predict_tavg=}, {predict_tstd=};"
                    f"In: {inlongest=}, {inshortest=}, inavg={inavg:.0f}, instd={instd:.0f};"
                    f"ACTUAL: {actual_sum=}, {actual_mean=}, {actual_max=}, {actual_min=}"
                )
            output = self.rollout_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output
