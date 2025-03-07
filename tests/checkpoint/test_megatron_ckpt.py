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
Using FSDPTrainer
"""
import os

import hydra
import ray
from transformers import AutoTokenizer

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.fs import copy_local_path_from_hdfs

MODEL_PATH = 'Qwen/Qwen2.5-0.5B'
DATA_PATH = 'data/gsm8k/'
SAVE_PATH = '/tmp/checkpoint'


def make_reward_function(tokenizer, num_examine):
    return None


additional_config = {
    'data': {
        'train_files': f'{DATA_PATH}/train.parquet',
        'val_files': f'{DATA_PATH}/test.parquet',
        'train_batch_size': 1024,
        'val_batch_size': 1312,
        'max_prompt_length': 512,
        'max_response_length': 512
    },
    'actor_rollout_ref': {
        'model': {
            'path': MODEL_PATH
        },
        'actor': {
            'optim': {
                'lr': 2e-6
            },
            'ppo_mini_batch_size': 32,
            'ppo_micro_batch_size_per_gpu': 1,
            'megatron': {
                'tensor_model_parallel_size': 2,
                'pipeline_model_parallel_size': 4,
            }
        },
        'rollout': {
            'log_prob_micro_batch_size_per_gpu': 8,
            'tensor_model_parallel_size': 2,
            'name': 'vllm',
            'gpu_memory_utilization': 0.5
        },
        'ref': {
            'log_prob_micro_batch_size_per_gpu': 16,
            'megatron': {
                'tensor_model_parallel_size': 2
            }
        }
    },
    'critic': {
        'optim': {
            'lr': 2e-5
        },
        'model': {
            'path': MODEL_PATH,
            'enable_gradient_checkpointing': False
        },
        'ppo_micro_batch_size_per_gpu': 4,
        'megatron': {
            'tensor_model_parallel_size': 2
        }
    },
    'algorithm': {
        'kl_ctrl': {
            'kl_coef': 0.001
        },
        'adv_estimator': 'grpo',
    },
    'trainer': {
        'critic_warmup': 0,
        'logger': ['console'],
        'project_name': 'verl_megatron_gsm8k_examples',
        'experiment_name': 'qwen2_5_0b5_function_rm',
        'n_gpus_per_node': 8,
        'nnodes': 1,
        'save_freq': 1,
        'test_freq': 1,
        'total_epochs': 15,
        'total_training_steps': 3,
    }
}


def check_result(origin_path, megatron_path, input_text):
    from transformers import AutoModelForCausalLM
    import torch
    print("check result")
    torch_dtype = torch.float16
    origin_model = AutoModelForCausalLM.from_pretrained(
        origin_path,
        torch_dtype=torch_dtype,
    ).eval()

    origin_model = origin_model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(origin_path)

    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
    origin_outputs = origin_model.generate(**inputs, max_new_tokens=8, do_sample=False)
    origin_text = tokenizer.decode(origin_outputs[0], skip_special_tokens=True)
    print(f"origin_text: {origin_text}")

    megatron_model = AutoModelForCausalLM.from_pretrained(
        megatron_path,
        torch_dtype=torch_dtype,
    ).eval()
    megatron_model = megatron_model.to('cuda')
    megatron_outputs = megatron_model.generate(**inputs, max_new_tokens=8, do_sample=False)
    megatron_text = tokenizer.decode(megatron_outputs[0], skip_special_tokens=True)
    print(f"megatron_text: {megatron_text}")

    assert origin_text == megatron_text, "megatron ckpt is diff from origin ckpt"


@hydra.main(config_path='../../../verl/verl/trainer/config', config_name='ppo_megatron_trainer', version_base=None)
def main(config):
    ray.init()

    from omegaconf import OmegaConf
    from pprint import pprint

    additional_omegaconf = OmegaConf.create(additional_config)
    config = OmegaConf.merge(config, additional_omegaconf)

    # print initial config
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values

    # print the config
    print('Config after normalizing batch_size')
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values

    config.trainer.logger = ['console']
    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    local_path = os.path.expanduser(local_path)
    # instantiate tokenizern
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    print(f'Tokenizer vocab_size: {tokenizer.vocab_size}')

    # define worker classes
    from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
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
    print(f"actor model : {trainer.actor_rollout_wg}")
    trainer.actor_rollout_wg.save_checkpoint(SAVE_PATH)


if __name__ == '__main__':
    main()
    check_result(MODEL_PATH, SAVE_PATH, "who are you？")
