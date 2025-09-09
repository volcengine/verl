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

import ray
from hydra import compose, initialize_config_dir
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from tests.workers.reward_model.agent_utils import init_agent_loop_manager
from verl.protocol import DataProto
from verl.trainer.main_ppo import create_rl_sampler
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def test_agent_loop_compute_score_with_model():
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        }
    )

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose("ppo_trainer")

    model_path = "meta-llama/Llama-3.2-3B-Instruct"
    rm_path = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    config.data.return_raw_chat = True
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.actor.use_dynamic_bsz = True
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.temperature = 0.0
    config.actor_rollout_ref.rollout.prompt_length = 1024
    config.actor_rollout_ref.rollout.response_length = 4096

    if os.environ["LEGACY_IMPL_RM"] == "disable":
        from verl.workers.config import HFModelConfig, RewardModelConfig

        model_config = HFModelConfig(path=rm_path)
        reward_model_config = RewardModelConfig(
            enable=True,
            enable_resource_pool=True,
            n_gpus_per_node=4,
            nnodes=1,
            model_config=model_config,
            input_model_config=None,
            tensor_model_parallel_size=2,
            gpu_memory_utilization=0.8,
        )
    else:
        config.reward_model.enable = True
        config.reward_model.model.path = rm_path
        config.reward_model.use_dynamic_bsz = True
        config.reward_model.forward_max_token_len_per_gpu = 6000
        config.reward_model.micro_batch_size_per_gpu = 40
        config.reward_model.enable_resource_pool = True
        config.reward_model.n_gpus_per_node = 4
        config.reward_model.nnodes = 1
        config.reward_model.model.trust_remote_code = True
        config.reward_model.model.input_tokenizer = None
        reward_model_config = None

    config.trainer.n_gpus_per_node = 4
    config.trainer.nnodes = 1
    # 1. init agent loop manager
    agent_loop_manager = init_agent_loop_manager(config, reward_model_config)

    # 2. init dataset and dataloader
    local_folder = os.path.expanduser("~/verl-data/gsm8k/")
    data_files = [os.path.join(local_folder, "train.parquet")]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = RLHFDataset(
        data_files=data_files,
        tokenizer=tokenizer,
        config=config.data,
        processor=None,
    )

    batch_size = 128
    sampler = create_rl_sampler(config.data, dataset)
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=config.data.dataloader_num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    # 3. generate_sequences with agent loop
    batch_dict = next(iter(dataloader))
    batch = DataProto.from_single_dict(batch_dict)
    gen_batch = agent_loop_manager.generate_sequences(prompts=batch)

    rm_scores = gen_batch.batch["rm_scores"]
    sample_scores = rm_scores.sum(dim=1)
    print(sample_scores)
    ray.shutdown()
