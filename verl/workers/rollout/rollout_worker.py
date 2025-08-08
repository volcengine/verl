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


from dataclasses import dataclass, field

import torch

import os
import datetime

from verl.single_controller.base import Worker
from verl import DataProto
import ray

from verl.single_controller.base.decorator import register, make_nd_compute_dataproto_dispatch_fn

from verl.base_config import BaseConfig

from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    is_cuda_available,
    is_npu_available,
)


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1



@dataclass
class EngineConfig(BaseConfig):
    pass


@dataclass
class vLLMEngineConfig(EngineConfig):
    swap_space: int = None
    disable_mm_preprocessor_cache: bool = True


@dataclass
class SGLangEngineConfig(EngineConfig):
    attention_backend: str = None


@dataclass
class MultiTurnConfig(BaseConfig):
    enable: bool = False
    max_assistant_turns: int = None
    tool_config_path: str = None
    max_user_turns: int = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    interaction_config_path: str = None
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    agent_loop_config_path: str = None



@dataclass
class RolloutConfig(BaseConfig):
    name: str
    mode: str = "sync"

    train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    val_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    prompt_length: int = 512
    response_length: int = 512
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    free_cache_engine: bool = True
    tensor_model_parallel_size: int = 2
    max_num_batched_tokens: int = 8192
    max_model_len: int = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the 
    log_prob_micro_batch_size_per_gpu: int = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384

    disable_log_stats: bool = True
    
    multi_stage_wake_up: bool = False
    engine_kwargs: EngineConfig = field(default_factory=EngineConfig)

    calculate_log_probs: bool = False
    update_weights_bucket_megabytes: int = 512



@ray.remote
class RolloutWorker(Worker):
    def __init__(self, config: RolloutConfig) -> None:
        super().__init__()
        self.config = config
        import torch.distributed

        self.device_name = get_device_name()

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{self.device_name}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            self.device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )

        rollout_name = self.config.name

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # build rollout engine here
        if self.config.name == "hf":
            pass
        elif self.config.name == "vllm":
            pass
        elif self.config.name == "sglang":
            pass

        
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer"))
    def generate_sequences(self, data: DataProto):
        """Given a batch of prompts, return a batch of responses. Internally, it can use 
        """
        pass

    
    



