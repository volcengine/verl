import time
from functools import wraps

import numpy as np
import pytest
import ray
import torch
from datasets import load_dataset
from omegaconf import OmegaConf

from tests.workers.rollout.async_rollout_utils import init_async_rollout_manager
from verl.protocol import DataProto

do_bench = True


def skip_if_false(exp):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not exp:
                pytest.skip("test will be skipped")

        return wrapper

    return decorator


class TestVllmMicroBatchScheduler:
    @pytest.fixture
    def ray_env(self):
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
        return runtime_env

    @pytest.fixture
    def aime_dataset(self):
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        prompts = DataProto(non_tensor_batch={"raw_prompt": np.array([[{"role": "user", "content": problem}] for problem in dataset["Problem"]])})
        return [prompts, dataset]

    @pytest.fixture
    def code_dataset(self):
        def gen_code():
            dataset = load_dataset("/demo-huabei2/chenhaiquan/dataset/Eurus-2-RL-Data/", split="train")
            prompts = DataProto(non_tensor_batch={"raw_prompt": np.array([dataset["prompt"][idx] for idx in range(256)])})
            return [prompts, dataset]

        return gen_code

    @pytest.fixture
    def small_model_path(self):
        return "Qwen/Qwen3-4B"

    @pytest.fixture
    def large_model_path(self):
        return "/demo-huabei2/common-models/Qwen/Qwen2.5-7B-Instruct"

    def test_micro_batch_scheduler(self, ray_env, aime_dataset, small_model_path):
        ray.init(
            runtime_env=ray_env,
        )
        # Load config
        config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
        config.actor_rollout_ref.model.path = small_model_path
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.MicroBatchChatCompletionScheduler"
        config.actor_rollout_ref.rollout.prompt_length = 8192
        config.actor_rollout_ref.rollout.response_length = 8192

        # Init sandbox and async rollout manager
        async_rollout_manager = init_async_rollout_manager(config, scheduler_kwargs={"max_inflight_req": 4})

        # Build dataset
        prompts, dataset = aime_dataset[0], aime_dataset[1]
        print(f"length of data proto : {len(prompts)}")
        start_time = time.time()
        micro_result = async_rollout_manager.generate_sequences(prompts=prompts)
        print(f"length of micro_result : {len(micro_result)}")
        print(f"time cost for micro_result : {time.time() - start_time}")
        torch.save(micro_result, "micro_result.pt")
        assert len(micro_result) == len(dataset)
        ray.timeline("micro_batch_scheduler.json")
        ray.shutdown()

        ray.init(
            runtime_env=ray_env,
        )

        config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler"

        # Init sandbox and async rollout manager
        async_rollout_manager = init_async_rollout_manager(config)
        start_time = time.time()
        native_result = async_rollout_manager.generate_sequences(prompts=prompts)
        print(f"length of native_result : {len(native_result)}")
        print(f"time cost for native_result : {time.time() - start_time}")
        torch.save(native_result, "native_result.pt")
        assert len(native_result) == len(dataset)
        ray.timeline("native_batch_scheduler.json")
        for i in range(len(dataset)):
            """
            a[0].__dict__['batch']
            TensorDict(
                fields={
                    attention_mask: Tensor(shape=torch.Size([8561]), device=cpu, dtype=torch.int64, is_shared=False),
                    input_ids: Tensor(shape=torch.Size([8561]), device=cpu, dtype=torch.int64, is_shared=False),
                    position_ids: Tensor(shape=torch.Size([8561]), device=cpu, dtype=torch.int64, is_shared=False),
                    prompts: Tensor(shape=torch.Size([363]), device=cpu, dtype=torch.int64, is_shared=False),
                    responses: Tensor(shape=torch.Size([8198]), device=cpu, dtype=torch.int64, is_shared=False)},
                batch_size=torch.Size([]),
                device=None,
                is_shared=False)
            """
            torch.allclose(micro_result[i].__dict__["batch"]["responses"], native_result[i].__dict__["batch"]["responses"])
            torch.allclose(micro_result[i].__dict__["batch"]["attention_mask"], native_result[i].__dict__["batch"]["attention_mask"])
            torch.allclose(micro_result[i].__dict__["batch"]["input_ids"], native_result[i].__dict__["batch"]["input_ids"])
            torch.allclose(micro_result[i].__dict__["batch"]["position_ids"], native_result[i].__dict__["batch"]["position_ids"])
            torch.allclose(micro_result[i].__dict__["batch"]["prompts"], native_result[i].__dict__["batch"]["prompts"])

    # @skip_if_false(True)
    def test_bench_micro_batch_scheduler(self, ray_env, code_dataset, large_model_path):
        if not do_bench:
            return
        ray.init(
            runtime_env=ray_env,
        )
        # Load config
        config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
        config.actor_rollout_ref.model.path = large_model_path
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.MicroBatchChatCompletionScheduler"
        config.actor_rollout_ref.rollout.prompt_length = 8192
        config.actor_rollout_ref.rollout.response_length = 8192
        # Init sandbox and async rollout manager
        async_rollout_manager = init_async_rollout_manager(config, scheduler_kwargs={"max_inflight_req": 32})
        # Build dataset
        prompts, dataset = code_dataset()
        print(f"length of data proto : {len(prompts)}")
        start_time = time.time()
        micro_result = async_rollout_manager.generate_sequences(prompts=prompts)
        print(f"length of micro_result : {len(micro_result)}")
        print(f"time cost for micro_result : {time.time() - start_time}")
        torch.save(micro_result, "micro_result.pt")
        assert len(micro_result) == len(prompts)
        ray.timeline("micro_batch_scheduler.json")
        ray.shutdown()

        ray.init(
            runtime_env=ray_env,
        )

        config.actor_rollout_ref.rollout.chat_scheduler = "examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler"

        # Init sandbox and async rollout manager
        async_rollout_manager = init_async_rollout_manager(config)
        start_time = time.time()
        native_result = async_rollout_manager.generate_sequences(prompts=prompts)
        print(f"length of native_result : {len(native_result)}")
        print(f"time cost for native_result : {time.time() - start_time}")
        torch.save(native_result, "native_result.pt")
        assert len(native_result) == len(prompts)
        ray.timeline("native_batch_scheduler.json")
