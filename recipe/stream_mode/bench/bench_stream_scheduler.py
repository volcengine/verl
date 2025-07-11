import os
import time

import ray
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from recipe.stream_mode.utils import init_async_rollout_manager
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
    return "/demo-huabei2/common-models/Qwen/Qwen2.5-7B-Instruct"
    # return "Qwen/Qwen3-8B"


def get_gsm8k_data():
    # prepare test dataset
    local_folder = os.path.expanduser("~/verl-data/gsm8k/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def get_dapo_data():
    local_folder = os.path.expanduser("/demo-huabei2/lusz/dataset/BytedTsinghua-SIA-AIME-2024/data/")
    local_path = os.path.join(local_folder, "aime-2024.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def get_code_data():
    local_folder = os.path.expanduser("/demo-huabei2/chenhaiquan/dataset/Eurus-2-RL-Data/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


ray.init(
    runtime_env=ray_env(),
)
os.environ["VERL_QUEUE_LOGGING_LEVEL"] = "INFO"
os.environ["VERL_LOGGING_LEVEL"] = "DEBUG"


# Load config
config = OmegaConf.load("recipe/stream_mode/config/stream_ppo_trainer.yaml")
config.trainer.n_gpus_per_node = 4
config.actor_rollout_ref.model.path = large_model_path()
config.actor_rollout_ref.rollout.mode = "async"
config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
config.actor_rollout_ref.rollout.chat_scheduler.micro_batch.max_inflight_req = 256
config.actor_rollout_ref.rollout.chat_scheduler.name = "stream"
config.actor_rollout_ref.rollout.mode = "async"
config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
config.actor_rollout_ref.rollout.prompt_length = 2048
config.actor_rollout_ref.rollout.response_length = 16384
config.actor_rollout_ref.rollout.temperature = 0
config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.5
config.actor_rollout_ref.rollout.repetition_penalty = 1.0
config.actor_rollout_ref.rollout.n = 1
config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
config.actor_rollout_ref.rollout.top_p = 0.6
config.actor_rollout_ref.rollout.top_k = -1
config.actor_rollout_ref.rollout.presence_penalty = 0.0
config.actor_rollout_ref.rollout.frequency_penalty = 0.0

tokenizer = hf_tokenizer(large_model_path())
# local_path = get_code_data()
local_path = get_gsm8k_data()
# local_path = get_dapo_data()
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
dataset.dataframe = dataset.dataframe.select(range(5000))
# # assert len(dataset) == 1000
print("dataset length", len(dataset))
batch_size = 1024
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)

# Init sandbox and async rollout manager
async_rollout_manager = init_async_rollout_manager(config)
start_time = time.time()
epoch_data = []
stop_epoch = False
total_gen_batch = []
native_epoch_times = []
native_batch_cost = []
print(f"length of data proto : {len(dataloader)}")
for _ in range(1):
    renew = True
    stop_epoch = False
    epoch_gen_batch = []
    # Build dataset
    # prompts, dataset = aime_dataset[0], aime_dataset[1]
    for batch_dict in dataloader:
        batch_time = time.time()
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        # pop those keys for generation
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
        print("all wake up")
        # this is for batch
        gen_batch_result = async_rollout_manager.generate_sequences(gen_batch)
        async_rollout_manager.sleep()
        print("sleep finished")
        epoch_data.append(gen_batch_result)
        epoch_gen_batch.append(gen_batch)
        batch_cost = time.time() - batch_time
        print(f"time cost for batch : {batch_cost}")
        native_batch_cost.append(batch_cost)

    cost = time.time() - start_time
    native_epoch_times.append(cost)
    print(f"time cost for epoch : {cost}")
    start_time = time.time()

ray.shutdown()


# print(f"time cost for each epoch for native scheduler : {native_epoch_times}")

ray.init(
    runtime_env=ray_env(),
)
os.environ["VERL_QUEUE_LOGGING_LEVEL"] = "INFO"
os.environ["VERL_LOGGING_LEVEL"] = "DEBUG"


# assert len(dataset) == 1000
print("dataset length", len(dataset))
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=collate_fn)
# Init sandbox and async rollout manager
async_rollout_manager = init_async_rollout_manager(config)
start_time = time.time()
epoch_data = []
stop_epoch = False
total_gen_batch = []
batch_time = []
for _ in range(1):
    renew = True
    stop_epoch = False
    epoch_gen_batch = []
    data_iter = iter(dataloader)
    epoch_times = []
    # Build dataset
    # prompts, dataset = aime_dataset[0], aime_dataset[1]
    while not stop_epoch:
        batch_start_time = time.time()
        print(f"length of data proto : {len(dataloader)}, renew: {renew}")
        async_rollout_manager.wake_up()
        print("all wake up")
        stop_epoch, gen_batch_result, gen_batch, batch = async_rollout_manager.stream_generate_sequences(
            data_iter, batch_size, renew
        )
        async_rollout_manager.sleep()
        print("sleep finished")
        epoch_data.append(gen_batch_result)
        epoch_gen_batch.append(gen_batch)
        renew = False
        batch_cost = time.time() - batch_start_time
        print(f"time cost for batch : {batch_cost}")
        batch_time.append(batch_cost)
    total_gen_batch.append(epoch_gen_batch)
    cost = time.time() - start_time
    epoch_times.append(cost)
    print(f"time cost for epoch : {cost}")
    start_time = time.time()

print(f"time cost for each epoch for stream scheduler : {epoch_times}")
print(f"time cost for each batch for stream scheduler : {batch_time}")

ray.shutdown()


print(f"time cost for each epoch for native scheduler : {native_epoch_times}")
print(f"time cost for each epoch for stream scheduler : {epoch_times}")
# compare:
print(f"stream cost time: {sum(epoch_times)}, \n native cost time: {sum(native_epoch_times)}")
print(f"stream cost time: {batch_time},\n native cost time: {native_batch_cost}")
