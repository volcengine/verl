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

import os
import time

import pytest
import ray
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from recipe.reorder_rollout.utils import init_async_rollout_manager


def get_gsm8k_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/data/gsm8k/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


class TestReorderScheduler:
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
    def small_model_path(self):
        return "Qwen/Qwen3-4B"

    @pytest.fixture
    def large_model_path(self):
        return "Qwen/Qwen2.5-7B-Instruct"

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
    def config(self):
        with initialize(version_base=None, config_path="../config"):
            cfg = compose(config_name="reorder_rollout_trainer")
            return cfg

    def test_reorder_scheduler(self, ray_env, large_model_path, config):
        ray.init(
            runtime_env=ray_env,
        )
        os.environ["VERL_QUEUE_LOGGING_LEVEL"] = "INFO"
        os.environ["VERL_LOGGING_LEVEL"] = "DEBUG"
        # Load config
        config.actor_rollout_ref.model.path = large_model_path
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.chat_scheduler.micro_batch.max_inflight_req = 2
        config.actor_rollout_ref.rollout.chat_scheduler.name = "reorder"
        config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
        config.actor_rollout_ref.rollout.prompt_length = 8192
        config.actor_rollout_ref.rollout.response_length = 1024
        config.actor_rollout_ref.rollout.temperature = 0.5
        config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.5
        config.data.train_batch_size = 3
        config.actor_rollout_ref.rollout.n = 5
        config.actor_rollout_ref.rollout.chat_scheduler.prefetch_factor = 2
        config.actor_rollout_ref.rollout.chat_scheduler.partial_policy = "keep"

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

        # Init async rollout manager
        async_rollout_manager = init_async_rollout_manager(config)
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
            while not stop_epoch:
                print(f"length of data proto : {len(dataloader)}, renew: {renew}")
                async_rollout_manager.wake_up()
                print("all wake up")
                stop_epoch, gen_batch_result, gen_batch, batch = async_rollout_manager.reorder_generate_sequences(
                    data_iter, renew=renew
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

        ray.shutdown()
