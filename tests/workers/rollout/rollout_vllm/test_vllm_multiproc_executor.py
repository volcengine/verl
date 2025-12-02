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

import os
from typing import Type

from tests.utils import multi_gpu_test
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.executor.executor_base import ExecutorBase
from vllm.utils import get_open_port
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.utils import get_loopback_ip, get_open_port

from verl.workers.rollout.vllm_rollout.vllm_multiproc_executor import vLLMMultiprocExecutor

MODEL = "facebook/opt-125m"
os.environ["VERL_VLLM_VOCAB_SIZE"] = "50272"


def create_vllm_config(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    max_model_len: int = 256,
    gpu_memory_utilization: float = 0.3,
    distributed_executor_backend: Type[ExecutorBase] = vLLMMultiprocExecutor,
    nnodes: int = 1,
    node_rank: int = 0,
    master_port: int = 0,
) -> VllmConfig:
    """Create a VllmConfig for testing using EngineArgs."""
    engine_args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        distributed_executor_backend=distributed_executor_backend,
        enforce_eager=True,
    )
    vllm_config = engine_args.create_engine_config()

    # Override distributed node settings if needed
    if nnodes > 1 or node_rank > 0:
        vllm_config.parallel_config.nnodes = nnodes
        vllm_config.parallel_config.node_rank = node_rank
        vllm_config.parallel_config.master_port = master_port
    if nnodes > 1:
        vllm_config.parallel_config.disable_custom_all_reduce = True

    return vllm_config


def create_test_scheduler_output(num_requests: int = 1) -> SchedulerOutput:
    """Create a minimal SchedulerOutput for testing."""
    # This is a simplified version - in practice you'd need proper
    # SchedulerOutput construction based on the actual vLLM v1 API
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_resumed_reqs=[],
        scheduled_running_reqs=[],
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
    )

def test_multiproc_executor_initialization():
    """Test that MultiprocExecutor can be initialized with proper config."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    # Create executor - this should initialize workers
    os.environ["VERL_VLLM_EXECUTOR_ZMQ_ADDRESS"] = f"tcp://{get_loopback_ip()}:{get_open_port()}"
    # os.environ["VERL_VLLM_FP8_QUANT_ENABLED"] = "1"
    executor = vLLMMultiprocExecutor(vllm_config=vllm_config)

    # Verify executor properties
    assert executor.world_size == 1, "World size should be 1 for single GPU"
    assert hasattr(executor, "workers"), "Executor should have workers"
    assert len(executor.workers) == 1, "Should have 1 worker for single GPU"

    # Clean up
    executor.shutdown()

@multi_gpu_test(num_gpus=2)
def test_multiproc_executor_initialization_tensor_parallel():
    """Test MultiprocExecutor initialization with tensor parallelism."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )

    # Create executor
    os.environ["VERL_VLLM_EXECUTOR_ZMQ_ADDRESS"] = f"tcp://{get_loopback_ip()}:{get_open_port()}"
    executor = vLLMMultiprocExecutor(vllm_config=vllm_config)

    # Verify executor properties
    assert executor.world_size == 2, "World size should be 2 for TP=2"
    # assert executor.local_world_size == 2, "Local world size should be 2"
    assert len(executor.workers) == 2, "Should have 2 workers for TP=2"

    # Verify output rank calculation
    output_rank = executor._get_output_rank()
    assert output_rank == 0, "Output rank should be 0 for TP=2, PP=1"

    # Clean up
    executor.shutdown()


@multi_gpu_test(num_gpus=2)
def test_multiproc_executor_collective_rpc():
    """Test collective RPC calls to all workers."""
    vllm_config = create_vllm_config(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )

    # Create executor
    os.environ["VERL_VLLM_EXECUTOR_ZMQ_ADDRESS"] = f"tcp://{get_loopback_ip()}:{get_open_port()}"
    executor = vLLMMultiprocExecutor(vllm_config=vllm_config)

    try:
        # Test check_health RPC - should work without errors
        executor.check_health()

        # Test that RPC works correctly
        # Note: We're just testing that the RPC mechanism works,
        # not testing actual model execution here
        assert not executor.is_failed, "Executor should not be in failed state"

    finally:
        # Clean up
        executor.shutdown()