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
"""
Using FSDPTrainer
"""
import re
import os
import hydra
import numpy as np
import ray
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from tests.e2e.envs.digit_completion import CharTokenizer
import pandas as pd


def make_reward_function(tokenizer, num_examine):

    def arithmetic_sequence_reward_function(data: DataProto):
        from tests.e2e.envs.digit_completion.task import compute_reward
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        for i in range(data.batch.batch_size[0]):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            # extract raw prompt
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # extract response
            response_ids = data_item.batch['responses']
            response_length = response_ids.shape[-1]
            response_mask = data.batch['attention_mask'][i][-response_length:]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt = tokenizer.decode(valid_prompt_ids)
            response = tokenizer.decode(valid_response_ids)
            # remove bos and eos
            prompt = prompt.replace(tokenizer.sep_token, '')
            response = response.replace(tokenizer.eos_token, '')
            if i < num_examine:
                print(prompt, response)

            reward_output = compute_reward(prompt, response)
            dense_reward = reward_output[0].tolist()
            ground_truth_response = reward_output[1]['ground_truth_response']
            if len(dense_reward) > 0:
                last_reward = dense_reward[-1]
            else:
                if len(ground_truth_response) == 0:
                    last_reward = 1
                else:
                    last_reward = 0

            # pad to response_length
            for _ in range(reward_tensor.shape[-1] - len(dense_reward)):
                dense_reward.append(last_reward)

            dense_reward = torch.as_tensor(dense_reward, dtype=torch.float32, device=reward_tensor.device)
            reward_tensor[i] = dense_reward * response_mask

        return reward_tensor

    return arithmetic_sequence_reward_function


@hydra.main(config_path='config', config_name='ray_trainer', version_base=None)
def main(config):
    ray.init(
        runtime_env={
            'env_vars': {
                'MEGATRON_USE_CUDA_TIMER': '0',
                'MEGATRON_START_PROCESS_TIMER': 'False',
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN'
            }
        })

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values

    dp_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    # normalize batch_size
    # TODO: move this inside each role
    config.actor_rollout_ref.actor.ppo_mini_batch_size //= dp_size
    config.actor_rollout_ref.actor.ppo_micro_batch_size //= dp_size
    config.critic.ppo_micro_batch_size //= dp_size
    config.actor_rollout_ref.rollout.micro_batch_size //= dp_size

    # print the config
    # print initial config
    print('Config after normalizing batch_size')
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.tokenizer_path)
    local_path = os.path.expanduser(local_path)
    # instantiate tokenizern
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    print(f'Tokenizer vocab_size: {tokenizer.vocab_size}')

    # define worker classes
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    reward_fn = make_reward_function(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            reward_fn=reward_fn,
                            val_reward_fn=reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
