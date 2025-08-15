# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
#* ray
python verl/tests/workers/rollout/rollout_vllm/test_vllm_random.py \
    -tp 1 \
    -dp 2 \
    -n 8 \
    --n_gpus_per_node 2 \
    --nnodes 1

#* 单机

python verl/tests/workers/rollout/rollout_vllm/test_vllm_random.py  \
    --dataset_path /sfs_turbo/wlf/VerlCode/dataset/dapo-math-17k.parquet \
    -tp 4 \
    -dp 4 \
    --gen_bs 64 \
    -n 8 \
    --n_gpus_per_node 16 \
    --nnodes 1


#* 单机减层
python verl/tests/workers/rollout/rollout_vllm/test_vllm_random.py \
    -tp 1 \
    -dp 1 \
    -n 4 \
    --n_gpus_per_node 1 \
    --dataset_path /sfs_turbo/wlf/VerlCode/dataset/dapo-math-17k.parquet \
    --load_format dummy \
    --nnodes 1 \
    --hdfs_path  /sfs_turbo/wlf/VerlCode/tmp/Qwen3-30B-A3B-No-think \
    --gen_bs 16 \
    --max_num_batched_tokens 512 \
    --max_num_seqs 16 \


# 单机完整 （实际运行请用这个）
python verl/tests/workers/rollout/rollout_vllm/test_vllm_random.py \
    -tp 4 \
    -dp 4 \
    -n 8 \
    --gen_bs 32 \
    --n_gpus_per_node 16 \
    --dataset_path /sfs_turbo/wlf/VerlCode/dataset/dapo-math-17k.parquet \
    --nnodes 1 \


python3 /sfs_turbo/wlf/VerlCode/dev/lhz/script/test_vllm_ray.py \
    -tp 2 \
    -dp 2 \
    -n 4 \
    --gen_bs 8 \
    --max_prompt_length $((2*1024)) \
    --max_response_length $((2*1024)) \
    --min_response_length $((1*1024)) \
    --max_num_batched_tokens $((22*1024)) \
    --n_gpus_per_node 8 \
    --dataset_path "/sfs_turbo/wlf/VerlCode/dataset/dapo-math-17k.parquet" \
    --hdfs_path "/sfs_turbo/pretrained_models/Qwen2.5-32B-Instruct" \
    --gpu_memory_utilization 0.90 \
    --nnodes 1 \

# 如果是 MoE 模型，需要开启 --enable_expert_parallel

"""

import argparse
import os
import socket
import time

import numpy as np
import ray
import rich
import torch
import torch.distributed as dist
import vllm.envs as envs
from tensordict import TensorDict
from vllm import LLM, SamplingParams

from verl.protocol import DataProto
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray import RayResourcePool
from verl.single_controller.ray.base import sort_placement_group_by_node_ip
from verl.utils.device import get_device_name, get_nccl_backend

# os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
# os.environ["TP_SOCKET_IFANAME"] = "eth0"
# os.environ["HCCL_SOCKET_IFNAME"] = "eth0"
# os.environ["ENABLE_MOE_ALLTOALLV"] = "1"
os.environ["VLLM_USE_V1"] = "1"

# * 这个必须加，不加会导致 set_device 报错
os.environ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("-dp", type=int, default=1)
parser.add_argument("-tp", type=int, default=1)
parser.add_argument("-n", type=int, default=4)
parser.add_argument("--gen_bs", type=int, default=128)
parser.add_argument("--n_gpus_per_node", type=int, default=8)
parser.add_argument("--nnodes", type=int, default=1)
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--enable_expert_parallel", action="store_true")
parser.add_argument("--load_format", type=str, default="safetensors", help="vllm读取权重的方式, dummy是随机初始化")

parser.add_argument("--hdfs_path", type=str, default="/path/to/model")
parser.add_argument("--max_num_batched_tokens", type=int, default=16 * 1024)
parser.add_argument("--max_num_seqs", type=int, default=1024)
parser.add_argument("--max_prompt_length", type=int, default=1024)
parser.add_argument("--max_response_length", type=int, default=1024)
parser.add_argument("--min_response_length", type=int, default=None)

parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)

args = parser.parse_args()

local_cache_path = "~/.cache/vllm"

WORLD_SIZE = args.nnodes * args.n_gpus_per_node
max_prompt_length = args.max_prompt_length
max_response_length = args.max_response_length
min_response_length = args.min_response_length
max_model_len = max_prompt_length + max_response_length
tp_size = int(args.tp)
dp_size = int(args.dp)
max_num_batched_tokens = args.max_num_batched_tokens  # debug
max_num_seqs = args.max_num_seqs
bs = args.gen_bs
# world_size = int(os.getenv("WORLD_SIZE", "-1"))

all_ranks = torch.arange(WORLD_SIZE).reshape(-1, dp_size, 1, tp_size)  # noqa
instance_size = WORLD_SIZE // all_ranks.shape[0]  # noqa

# * all_ranks.shape[0] 实例个数


def check_args():
    # * check gbs
    assert bs % (all_ranks.shape[0] * dp_size) == 0, "样本数必须是 实例数 * dp_size 的整数倍"
    pass


if args.dataset_path is None:
    preencode_prompts = [
        "Who won the Champions League in 2019?",
        "The founder of Apple is",
        "What's your name?",
        "What can Vscode do?",
        "Who is the best player in game CS?",
        "Why I could'nt connect github in China?",
    ]

else:
    from datasets import load_dataset

    if args.dataset_path.endswith("parquet"):
        data = load_dataset("parquet", data_files=args.dataset_path)["train"]
    else:
        raise TypeError()

    assert len(data) >= bs
    preencode_prompts = list(map(lambda x: x[0]["content"], data[:bs]["prompt"]))
assert len(preencode_prompts) % bs == 0, "Prompt batch size must be divisible by gen_bs"


def get_cluster_info():
    # 确保分布式环境已初始化
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment not initialized")

    world_size = dist.get_world_size()

    # 获取当前节点的IP地址
    ip_address = _get_current_node_ip()

    # 收集所有rank的IP地址
    ip_list = [None] * world_size
    dist.all_gather_object(ip_list, ip_address)

    return ip_list


def get_availale_master_addr_port():
    host_ip_by_sdk = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        free_port = sock.getsockname()[1]

    return host_ip_by_sdk, free_port


def _get_current_node_ip() -> str:
    # 创建一个 UDP 套接字（仅用于获取接口信息）
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        # 连接到一个外部地址（无需真实通信）
        s.connect(("8.8.8.8", 80))  # Google DNS 服务器
        local_ip = s.getsockname()[0]

    return local_ip


def _init_dp_envs():
    rank = torch.distributed.get_rank()
    world_size = int(os.getenv("WORLD_SIZE", "-1"))

    for index, group_rank in enumerate(all_ranks):
        if torch.distributed.get_rank() in group_rank:
            os.environ["VLLM_INSTANCE_INDEX"] = str(index)

    group_ranks = all_ranks.transpose(1, 3).reshape(-1, dp_size).unbind(0)

    # group_ranks = [x.tolist() for x in group_ranks]
    ip_list = get_cluster_info()
    for index, group_rank in enumerate(group_ranks):
        _group_rank = group_rank.tolist()
        if torch.distributed.get_rank() in _group_rank:
            os.environ["VLLM_DP_MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT")) + 1 + index)
            os.environ["VLLM_DP_MASTER_IP"] = ip_list[_group_rank[0]]

    local_dp_rank = rank // tp_size % dp_size
    os.environ["VLLM_DP_RANK"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_PORT"] = os.environ["VLLM_DP_MASTER_PORT"]
    envs.VLLM_DP_RANK = int(os.environ["VLLM_DP_RANK"])
    envs.VLLM_DP_MASTER_IP = os.environ["VLLM_DP_MASTER_IP"]
    envs.VLLM_DP_MASTER_PORT = int(os.environ["VLLM_DP_MASTER_PORT"])

    print(f"[VLLM] using {world_size=}, TP={tp_size}, DP={dp_size}", flush=True)


@ray.remote
class Vllm_Worker(Worker):
    def __init__(self, rank_zero_info):
        os.environ["WG_BACKEND"] = "ray"
        if rank_zero_info is None:
            raise RuntimeError()
        super().__init__()

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        self.rank_zero_info = rank_zero_info

        self.create_llm()

    def create_llm(self):
        from verl.utils.fs import copy_to_local

        self.local_model_path = copy_to_local(src=args.hdfs_path, cache_dir=local_cache_path)

        _init_dp_envs()
        self.dp_rank = int(os.environ.get("VLLM_DP_RANK", 0))
        self.dp_size = int(os.environ.get("VLLM_DP_SIZE", 1))
        self.vllm_instance_index = int(os.environ.get("VLLM_INSTANCE_INDEX", 1))

        rich.print("[green]构建LLM中[/green]")
        self.llm = LLM(
            model=self.local_model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=args.tp,
            distributed_executor_backend="external_launcher",
            dtype="bfloat16",
            enforce_eager=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            load_format=args.load_format,  #! 如果是减层
            disable_log_stats=False,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            trust_remote_code=True,
            enable_expert_parallel=args.enable_expert_parallel,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            seed=0,
        )
        if args.n == 1:
            temperature = 0.0
        else:
            temperature = 1.0
        top_p = 1
        top_k = -1
        kwargs = dict(
            n=args.n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,  # -1 for vllm rollout
            max_tokens=max_response_length,
            logprobs=1,
        )
        if args.min_response_length is not None:
            kwargs["min_tokens"] = args.min_response_length
            # kwargs['ignore_eos'] = True

        self.sampling_params = SamplingParams(**kwargs)
        self.sampling_params.detokenize = True

        rich.print("[green]构建LLM完毕[/green]")

    def dp_dispatch(self, prompt_batch: list):
        # *  根据实例数进行分配
        dispatch_prompt_batch = []

        if self.dp_size == 1:
            dispatch_prompt_batch = prompt_batch
        else:
            ind = torch.arange(len(prompt_batch)).reshape(-1, self.dp_size)[self.dp_rank]

            dispatch_prompt_batch = [prompt_batch[i] for i in ind.tolist()]

        return dispatch_prompt_batch

    def run(self, prompt_batch) -> DataProto:
        prompt_batch = self.dp_dispatch(list(map(str, prompt_batch.batch["prompt"])))

        rich.print("[green]Rollout[/green]")
        start_time = time.time()

        outputs = self.llm.generate(
            prompt_batch,
            sampling_params=self.sampling_params,
            use_tqdm=True,
        )

        cost_time = time.time() - start_time
        _output_dataproto = DataProto(
            TensorDict(
                {"prompt": prompt_batch, "response": outputs},
                batch_size=len(prompt_batch),
            )
        )
        _output_dataproto.meta_info = {"cost_time": cost_time}
        self.calc_tps(outputs, _output_dataproto.meta_info)
        # * 计算吞吐

        return _output_dataproto

    def calc_tps(self, outputs: list, meta_info: dict) -> dict:
        num_prompt = 0
        num_response = 0
        for request in outputs:
            num_prompt += len(request.prompt_token_ids)
            for resp_i in request.outputs:
                num_response += len(resp_i.token_ids)

        meta_info["tokens"] = num_prompt + num_response
        meta_info["tps"] = meta_info["tokens"] / meta_info["cost_time"]
        meta_info["mean_response"] = np.mean(num_response)

        return meta_info

    def get_attr(self, key):
        return getattr(self, key)


def test_vllm_spmd_ray():
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                    # "RAY_DEBUG_POST_MORTEM": "1",
                }
            },
        )

    # todo 目前只能单机内跑
    n_gpus_per_node = int(args.n_gpus_per_node)
    nnodes = int(args.nnodes)
    resource_pool = RayResourcePool(process_on_nodes=[n_gpus_per_node] * nnodes)

    strategy = "PACK"
    pgs = resource_pool.get_placement_groups(strategy=strategy, device_name="npu")
    world_size = resource_pool.world_size
    local_world_size = resource_pool.store[0]

    pg = placement_group([{"CPU": world_size * 8, "NPU": world_size}])

    rank = -1
    tasks = []
    master_addr = None
    master_port = None

    for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
        for local_rank in range(local_world_size):
            rank += 1
            options = {}

            env_vars = {
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
                "WG_BACKEND": "ray",
                "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
                "RAY_LOCAL_RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
            }

            actor_name = f"test_vllm_ray:{local_rank}"  # e.g. test_vllm_ray:5

            if rank == 0:
                master_addr, master_port = get_availale_master_addr_port()
                info = {
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": str(master_port),
                }

            env_vars["MASTER_ADDR"] = master_addr
            env_vars["MASTER_PORT"] = str(master_port)

            options = {
                "runtime_env": {"env_vars": env_vars},
                "resources": {"NPU": 1},
            }

            options.update(
                {
                    "scheduling_strategy": PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=local_rank,
                    ),
                    "name": actor_name,
                }
            )

            # * 这里必须得放一个被ray.remote装饰的类实例，不能是一个函数
            task = Vllm_Worker.options(**options).remote(info)
            tasks.append(task)

    # * start running
    tasklist = []
    _preencode_prompts = np.array(preencode_prompts, dtype=object).reshape(len(all_ranks), -1)

    for i, task_i in enumerate(tasks):
        ii = i // instance_size
        _prompt_ii = _preencode_prompts[ii]
        _input = DataProto(TensorDict({"prompt": list(map(str, _prompt_ii))}, batch_size=len(_prompt_ii)))
        tasklist.append(task_i.run.remote(_input))
    list_output = ray.get(tasklist)

    try:
        _output = list_output[0].batch["response"][0].outputs[0]
        response_text = _output.text
        print("===>Output===>", flush=True)
        if len(response_text) <= 620:
            print(response_text, flush=True)
        else:
            print(response_text[:300], flush=True)
            print("\n...\n...\n")
            print(response_text[-300:], flush=True)
        print(f"<===END, 生成结束原因: {_output.finish_reason}", flush=True)

    except Exception as e:
        print(f"Print generation failed! \nreason is {e.__repr__()}")

    # * 打印 综合 TPS
    tps = np.mean(list(map(lambda x: x.meta_info["tps"], list_output)))
    cost_time = np.mean(list(map(lambda x: x.meta_info["cost_time"], list_output)))
    print(f"TPS: {tps: 0.4f} tokens/s")
    print(f"平均耗时: {cost_time: 0.2f} s")


if __name__ == "__main__":
    check_args()
    test_vllm_spmd_ray()
