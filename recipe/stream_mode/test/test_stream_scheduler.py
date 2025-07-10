import os
import time
from functools import wraps

import numpy as np
import pytest
import ray
from datasets import load_dataset
from omegaconf import OmegaConf
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice
from torch.utils.data import DataLoader

from recipe.stream_mode.utils import init_async_rollout_manager
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


def get_gsm8k_data():
    # prepare test dataset
    local_folder = os.path.expanduser("/demo-huabei2/lusz/dataset/gsm8k/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def get_code_data():
    local_folder = os.path.expanduser("/demo-huabei2/chenhaiquan/dataset/Eurus-2-RL-Data/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


class TestStreamScheduler:
    @pytest.fixture
    def ray_env(self):
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "INFO",
                "VLLM_LOGGING_LEVEL": "DEBUG",
                "VLLM_USE_V1": "1",
                "VERL_LOGGING_LEVEL": "DEBUG",
                "VERL_QUEUE_LOGGING_LEVEL": "DEBUG",
            }
        }
        return runtime_env

    @pytest.fixture
    def aime_dataset(self):
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        prompts = DataProto(
            non_tensor_batch={
                "raw_prompt": np.array([[{"role": "user", "content": problem}] for problem in dataset["Problem"]])
            }
        )
        return [prompts, dataset]

    @pytest.fixture
    def code_dataset(self):
        def gen_code():
            dataset = load_dataset("/demo-huabei2/chenhaiquan/dataset/Eurus-2-RL-Data/", split="train")
            print("finish")
            prompts = DataProto(
                non_tensor_batch={"raw_prompt": np.array([dataset[idx]["prompt"] for idx in range(256, 256 + 8192)])}
            )
            return [prompts, dataset]

        return gen_code

    @pytest.fixture
    def small_model_path(self):
        return "Qwen/Qwen3-4B"

    @pytest.fixture
    def large_model_path(self):
        return "/demo-huabei2/common-models/Qwen/Qwen2.5-7B-Instruct"

    @pytest.fixture
    def sampling_params(self):
        return dict(
            n=1,
            temperature=0,
            top_p=1,
            top_k=-1,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            ignore_eos=False,
        )

    @pytest.fixture
    def chat_completion(self):
        return ChatCompletion(
            id="40359df5b352b344bc8e",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="The answer is 42.",
                    ),
                )
            ],
            created=1690917600,
            model="gpt-3.5-turbo",
            object="chat.completion",
        )

    @pytest.fixture
    def config(self):
        return OmegaConf.load("recipe/stream_mode/config/stream_ppo_trainer.yaml")

    @pytest.fixture
    def rollout_config(self):
        config = OmegaConf.load("recipe/stream_mode/config/stream_ppo_trainer.yaml")
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.chat_scheduler.micro_batch.max_inflight_req = 8
        config.actor_rollout_ref.rollout.chat_scheduler.name = "micro_batch"
        config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
        config.actor_rollout_ref.rollout.prompt_length = 8192
        config.actor_rollout_ref.rollout.response_length = 2048
        config.actor_rollout_ref.rollout.temperature = 0.0
        config.actor_rollout_ref.rollout.repetition_penalty = 1.0
        return config

    def test_streaming_scheduler(self, ray_env, aime_dataset, large_model_path, config):
        ray.init(
            runtime_env=ray_env,
        )
        os.environ["VERL_QUEUE_LOGGING_LEVEL"] = "INFO"
        os.environ["VERL_LOGGING_LEVEL"] = "DEBUG"
        # Load config
        config.actor_rollout_ref.model.path = large_model_path
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.chat_scheduler.micro_batch.max_inflight_req = 2
        config.actor_rollout_ref.rollout.chat_scheduler.name = "stream"
        config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
        config.actor_rollout_ref.rollout.prompt_length = 8192
        config.actor_rollout_ref.rollout.response_length = 1024
        config.actor_rollout_ref.rollout.temperature = 0.5
        config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.5
        config.actor_rollout_ref.rollout.repetition_penalty = 1.0
        config.actor_rollout_ref.rollout.n = 2

        from verl.utils import hf_tokenizer
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        tokenizer = hf_tokenizer("deepseek-ai/deepseek-coder-1.3b-instruct")
        local_path = get_gsm8k_data()
        data_config = OmegaConf.create(
            {
                "prompt_key": "prompt",
                "max_prompt_length": 8192,
                "filter_overlong_prompts": True,
                "filter_overlong_prompts_workers": 2,
                "return_raw_chat": True,
            }
        )
        dataset = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=data_config)

        dataset.dataframe = dataset.dataframe.select(range(10))
        assert len(dataset) == 10
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=collate_fn)

        # Init sandbox and async rollout manager
        async_rollout_manager = init_async_rollout_manager(config)
        batch_size = 3
        start_time = time.time()
        epoch_data = []
        stop_epoch = False
        total_gen_batch = []
        for _ in range(2):
            renew = True
            stop_epoch = False
            epoch_gen_batch = []
            data_iter = iter(dataloader)
            epoch_times = []
            # Build dataset
            # prompts, dataset = aime_dataset[0], aime_dataset[1]
            while not stop_epoch:
                print(f"length of data proto : {len(dataloader)}, renew: {renew}")
                async_rollout_manager.wake_up()
                print("all wake up")
                stop_epoch, gen_batch_result, gen_batch, batch = async_rollout_manager.stream_generate_sequences(
                    data_iter, batch_size, renew=renew
                )
                async_rollout_manager.sleep()
                print("sleep finished")
                epoch_data.append(gen_batch_result)
                epoch_gen_batch.append(gen_batch)
                renew = False
            total_gen_batch.append(epoch_gen_batch)
            cost = time.time() - start_time
            epoch_times.append(cost)
            print(f"time cost for batch : {cost}")
            start_time = time.time()

        assert len(total_gen_batch) == 2
        expect_length = [[3, 3, 3, 1], [3, 3, 3, 1]]
        for i in range(2):
            for j in range(4):
                assert len(total_gen_batch[i][j]) == expect_length[i][j]

        # torch.save(micro_result, "micro_result.pt")
        # assert len(gen_batch_result) == len(gen_batch)*config.actor_rollout_ref.rollout.n
        # assert len(gen_batch) == len(batch)
        # assert stop_epoch is False
        # then we need to verify the micro data order valid
        ray.shutdown()
