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

import hydra
import ray
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from recipe.reorder_rollout.utils import init_async_rollout_manager
from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def ray_env():
    runtime_env = {
        "env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "TRACE",
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "VLLM_USE_V1": "1",
            "VERL_LOGGING_LEVEL": "DEBUG",
            "VERL_QUEUE_LOGGING_LEVEL": "DEBUG",
        }
    }
    return runtime_env


def large_model_path():
    return "Qwen/Qwen2.5-7B-Instruct"


def get_gsm8k_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/verl-data/gsm8k/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def get_dapo_data():
    local_folder = os.path.expanduser("~/BytedTsinghua-SIA-AIME-2024/data/")
    local_path = os.path.join(local_folder, "aime-2024.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def get_code_data():
    local_folder = os.path.expanduser("~/Eurus-2-RL-Data/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def setup_environment():
    os.environ["VERL_QUEUE_LOGGING_LEVEL"] = "INFO"
    os.environ["VERL_LOGGING_LEVEL"] = "DEBUG"


def load_dataset(data_source="code", limit=5000):
    tokenizer = hf_tokenizer(large_model_path())
    if data_source == "gsm8k":
        local_path = get_gsm8k_data()
    elif data_source == "dapo":
        local_path = get_dapo_data()
    else:
        local_path = get_code_data()
    data_config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 2048,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
            "return_raw_chat": True,
        }
    )
    dataset = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=data_config)
    dataset.dataframe = dataset.dataframe.select(range(limit))
    return dataset


def run_native_scheduler(config, dataset, batch_size):
    setup_environment()
    ray.init(runtime_env=ray_env())
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn
    )
    async_rollout_manager = init_async_rollout_manager(config)
    native_batch_cost, native_epoch_times = [], []
    start_time = time.time()
    for _ in range(1):
        for batch_dict in dataloader:
            batch_time = time.time()
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            gen_batch = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            async_rollout_manager.wake_up()
            _ = async_rollout_manager.generate_sequences(gen_batch)
            async_rollout_manager.sleep()
            native_batch_cost.append(time.time() - batch_time)
        native_epoch_times.append(time.time() - start_time)
        start_time = time.time()
    ray.shutdown()
    print(f"Native epoch times: {native_epoch_times}")
    print(f"Native batch times: {native_batch_cost}")
    return native_epoch_times, native_batch_cost


def run_reorder_scheduler(config, dataset):
    setup_environment()
    ray.init(runtime_env=ray_env())
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=collate_fn)
    async_rollout_manager = init_async_rollout_manager(config)
    batch_time = []
    epoch_times = []
    start_time = time.time()
    for _ in range(1):
        renew = True
        stop_epoch = False
        epoch_gen_batch = []
        data_iter = iter(dataloader)
        while not stop_epoch:
            batch_start_time = time.time()
            print(f"length of data proto : {len(dataloader)}, renew: {renew}")
            async_rollout_manager.wake_up()
            print("all wake up")
            stop_epoch, gen_batch_result, gen_batch, batch = async_rollout_manager.reorder_generate_sequences(
                data_iter, renew
            )
            async_rollout_manager.sleep()
            print("sleep finished")
            epoch_gen_batch.append(gen_batch)
            renew = False
            batch_cost = time.time() - batch_start_time
            print(f"time cost for batch : {batch_cost}")
            batch_time.append(batch_cost)
        epoch_times.append(time.time() - start_time)
        start_time = time.time()
    ray.shutdown()
    print(f"Stream scheduler epoch times: {epoch_times}")
    print(f"Stream scheduler batch times: {batch_time}")
    return epoch_times, batch_time


def run_sync_batch_scheduler(config, dataset):
    setup_environment()
    ray.init(runtime_env=ray_env())
    config.actor_rollout_ref.rollout.chat_scheduler.synchronize_interval = 2
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=collate_fn)
    async_rollout_manager = init_async_rollout_manager(config)
    sync_batch_time = []
    sync_epoch_times = []
    start_time = time.time()
    for _ in range(1):
        renew = True
        stop_epoch = False
        epoch_gen_batch = []
        data_iter = iter(dataloader)
        while not stop_epoch:
            batch_start_time = time.time()
            print(f"length of data proto : {len(dataloader)}, renew: {renew}")
            async_rollout_manager.wake_up()
            print("all wake up")
            stop_epoch, gen_batch_result, gen_batch, batch = async_rollout_manager.reorder_generate_sequences(
                data_iter, renew
            )
            async_rollout_manager.sleep()
            print("sleep finished")
            epoch_gen_batch.append(gen_batch)
            renew = False
            batch_cost = time.time() - batch_start_time
            print(f"time cost for batch : {batch_cost}")
            sync_batch_time.append(batch_cost)
        sync_epoch_times.append(time.time() - start_time)
        start_time = time.time()
    ray.shutdown()
    print(f"Sync batch mode epoch times: {sync_epoch_times}")
    print(f"Sync batch mode batch times: {sync_batch_time}")
    return sync_epoch_times, sync_batch_time


@hydra.main(config_path="../config", config_name="reorder_rollout_trainer", version_base=None)
def main(cfg):
    cfg.trainer.n_gpus_per_node = 4
    cfg.actor_rollout_ref.model.path = large_model_path()
    cfg.actor_rollout_ref.rollout.mode = "async"
    cfg.actor_rollout_ref.rollout.multi_turn.format = "hermes"
    cfg.actor_rollout_ref.rollout.chat_scheduler.micro_batch.max_inflight_req = 288
    cfg.actor_rollout_ref.rollout.chat_scheduler.name = "reorder"
    cfg.actor_rollout_ref.rollout.prompt_length = 2048
    cfg.actor_rollout_ref.rollout.response_length = 16384
    cfg.actor_rollout_ref.rollout.temperature = 0
    cfg.actor_rollout_ref.rollout.gpu_memory_utilization = 0.6
    cfg.actor_rollout_ref.rollout.n = 5
    cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    cfg.actor_rollout_ref.rollout.stat_log_interval = 10
    cfg.actor_rollout_ref.rollout.top_p = 0.6
    cfg.actor_rollout_ref.rollout.top_k = -1
    cfg.actor_rollout_ref.rollout.chat_scheduler.prefetch_factor = 1.5
    cfg.actor_rollout_ref.rollout.chat_scheduler.partial_policy = "keep"

    batch_size = 1024
    cfg.data.train_batch_size = batch_size
    dataset = load_dataset(data_source="gsm8k", limit=5000)

    mode = "use_reorder"
    if mode == "use_native":
        run_native_scheduler(cfg, dataset, batch_size)
    elif mode == "use_reorder":
        run_reorder_scheduler(cfg, dataset)
    elif mode == "use_sync":
        run_sync_batch_scheduler(cfg, dataset)


if __name__ == "__main__":
    main()
