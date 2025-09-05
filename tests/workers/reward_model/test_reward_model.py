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
import torch
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role


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

    rm_path = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"

    if os.environ["LEGACY_IMPL_RM"] == "disable":
        from verl.workers.config import HFModelConfig, RewardModelConfig
        from verl.workers.roles import RewardModelWorker

        model_config = HFModelConfig(path=rm_path)
        reward_model_config = RewardModelConfig(
            enable=True,
            model_config=model_config,
            input_model_config=None,
            tensor_model_parallel_size=1,
            gpu_memory_utilization=0.8,
        )
    else:
        from verl.workers.fsdp_workers import RewardModelWorker

        config.reward_model.enable = True
        config.reward_model.model.path = rm_path
        config.reward_model.use_dynamic_bsz = True
        config.reward_model.forward_max_token_len_per_gpu = 6000
        config.reward_model.micro_batch_size_per_gpu = 40
        config.reward_model.model.trust_remote_code = True
        config.reward_model.model.input_tokenizer = None
        reward_model_config = config.reward_model

    config.trainer.n_gpus_per_node = 2
    config.trainer.nnodes = 1

    role_worker_mapping = {}
    if reward_model_config.enable:
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {}
    mapping[Role.RewardModel] = "global_pool"
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    if reward_model_config.enable:
        # we create a RM here
        resource_pool = resource_pool_manager.get_resource_pool(Role.RewardModel)
        rm_cls = RayClassWithInitArgs(role_worker_mapping[Role.RewardModel], config=reward_model_config)
        resource_pool_to_cls[resource_pool]["rm"] = rm_cls

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)

    rm_wg = all_wg["rm"]
    rm_wg.init_model()

    convs = [
        [
            {
                "role": "user",
                "content": "What is the range of the numeric output of a sigmoid node in a neural network?",
            },
            {"role": "assistant", "content": "The output is bounded between -1 and 1."},
        ],
        [
            {
                "role": "user",
                "content": "What is the range of the numeric output of a sigmoid node in a neural network?",
            },
            {"role": "assistant", "content": "The output is bounded between 0 and 1."},
        ],
    ]
    tokenizer = AutoTokenizer.from_pretrained(rm_path)

    prompt_length, response_length = 1024, 4096
    pad_token_id = tokenizer.pad_token_id
    prompts, responses, input_ids, attention_masks, position_ids = [], [], [], [], []
    for conv in convs:
        prompt = tokenizer.apply_chat_template(conv[:1], tokenize=True)
        response = tokenizer.apply_chat_template(conv, tokenize=True)[len(prompt) :]
        attention_mask = (
            [0] * (prompt_length - len(prompt))
            + [1] * len(prompt)
            + [1] * len(response)
            + [0] * (response_length - len(response))
        )
        prompt = [pad_token_id] * (prompt_length - len(prompt)) + prompt
        response = response + [pad_token_id] * (response_length - len(response))
        prompts.append(torch.tensor(prompt))
        responses.append(torch.tensor(response))
        input_ids.append(torch.tensor(prompt + response))
        attention_masks.append(torch.tensor(attention_mask))

    from verl.utils.model import compute_position_id_with_mask

    prompts = torch.stack(prompts)
    responses = torch.stack(responses)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    position_ids = compute_position_id_with_mask(attention_masks)
    data = DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "position_ids": position_ids,
        },
    )
    gen_batch = rm_wg.compute_rm_score(data)
    rm_scores = gen_batch.batch["rm_scores"]
    sample_scores = rm_scores.sum(dim=1)
    print(sample_scores)
    ray.shutdown()
