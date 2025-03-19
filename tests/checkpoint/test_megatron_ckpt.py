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
from os.path import expanduser
import torch
import torch.multiprocessing as mp

import hydra
import ray
from transformers import AutoTokenizer

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.megatron_utils import get_model_checkpoint_path
from multiprocessing import Process, log_to_stderr
import logging

MODEL_PATHS = ['Qwen/Qwen2.5-0.5B', 'deepseek-ai/deepseek-coder-1.3b-instruct']
MODEL_PATH = ''
DATA_PATH = expanduser('~/data/gsm8k')
SAVE_PATH = '/tmp/checkpoint'

additional_config = {}

def build_additional_configs(MODEL_PATH):
    global additional_config
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
                },
                'checkpoint_contents': ['model']
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
            },
            'checkpoint_contents': ['model']
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
            'total_training_steps': 1
        }
    }

def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)

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
        get_model_checkpoint_path(megatron_path),
        torch_dtype=torch_dtype,
    ).eval()
    megatron_model = megatron_model.to('cuda')
    megatron_outputs = megatron_model.generate(**inputs, max_new_tokens=8, do_sample=False)
    megatron_text = tokenizer.decode(megatron_outputs[0], skip_special_tokens=True)
    print(f"megatron_text: {megatron_text}")

    assert origin_text == megatron_text, "megatron ckpt is diff from origin ckpt"
    print("Checkpoint save/load test passed!")


@hydra.main(config_path='verl/trainer/config', config_name='ppo_megatron_trainer', version_base=None)
def main(config):
    from omegaconf import OmegaConf
    from pprint import pprint

    global additional_config
    print(f'MODEL_PATH: {MODEL_PATH}, additional_config: {additional_config}')
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
    print(f'local_path: {local_path}')
    # instantiate tokenizern
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    print(f'Tokenizer vocab_size: {tokenizer.vocab_size}')

    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    # define worker classes
    from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
    from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
    ray_worker_group_cls = NVMegatronRayWorkerGroup
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

    from verl.workers.reward_manager import NaiveRewardManager
    reward_manager_cls = NaiveRewardManager
        
    compute_score = get_custom_reward_fn(config)
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    # trainer.fit()
    trainer.actor_rollout_wg.save_checkpoint(SAVE_PATH)

def run_single_model(model_path):
    ray.init()
    global MODEL_PATH
    MODEL_PATH = model_path
    build_additional_configs(model_path)
    main()
    check_result(model_path, SAVE_PATH, "who are you？")

log_to_stderr(logging.DEBUG)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    for model_path in MODEL_PATHS:
        p = Process(target=run_single_model, args=(model_path,))
        p.start()
        p.join()
        p.terminate()
