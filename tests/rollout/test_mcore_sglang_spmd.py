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
import torch
import transformers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType

from verl.utils.model import update_model_config
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from verl.utils.debug import log_gpu_memory_usage
from transformers import GenerationConfig
from verl.utils.distributed import initialize_global_process_group
from verl.utils.torch_functional import pad_sequence_to_length
from megatron.core import parallel_state as mpu
from verl.utils.model import normalize_pp_vpp_params
from verl.utils.megatron.optimizer import get_megatron_optimizer
from megatron.core.models.gpt.gpt_model import ModelType
from verl.utils.model import print_model_size, update_model_config, get_generation_config
from verl.utils.megatron_utils import get_model, init_megatron_optim_config, convert_config
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer
from verl.utils.model import load_megatron_model_weights, load_megatron_gptmodel_weights
from verl.workers.sharding_manager import AllGatherPPModel
from vllm import LLM, SamplingParams
from verl.utils.megatron_utils import mcore_model_parallel_config
from verl.workers.sharding_manager import MegatronVLLMShardingManager
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode


def set_random_seed(seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.device_count() > 0:
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(seed)


def test_megatron_vllm():
    if not torch.distributed.is_initialized():
        rank = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)

        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=None,
            pipeline_model_parallel_split_rank=None,
            use_sharp=False,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            nccl_communicator_config_path=None,
        )

    set_random_seed(seed=1)

    # Step 1: initialize the tokenizer
    local_cache_path = '~/.cache/verl/rlhf'
    local_cache_path = os.path.expanduser(local_cache_path)
    hdfs_path = 'Qwen/Qwen2-7B-Instruct'
    from verl.utils.fs import copy_to_local
    local_path = copy_to_local(src=hdfs_path, cache_dir=local_cache_path)
    tokenizer = hf_tokenizer(local_path)

    # Step 2: get the actor_model_config
    actor_model_config = AutoConfig.from_pretrained(local_path)
    override_model_config = {}
    override_config_kwargs = {
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
    }
    override_config_kwargs.update(override_model_config)
    update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
    print(f'Model config after override: {actor_model_config}')

    megatron_config = mcore_model_parallel_config(sequence_parallel=True, params_dtype=torch.bfloat16)

    share_embeddings_and_output_weights = getattr(actor_model_config, "tie_word_embeddings", False)
    tfconfig = convert_config(actor_model_config, megatron_config)

    def megatron_actor_model_provider(pre_process, post_process):
        from verl.utils.model import get_parallel_gptmodel_from_config
        parallel_model = get_parallel_gptmodel_from_config(
            tfconfig,
            actor_model_config,
            pre_process,
            post_process,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            value=False)
        parallel_model.cuda()
        return parallel_model

    # Step 3: initialize the megatron model
    # Initialize the 3D HybridEngine
    hybrid_engine = AllGatherPPModel(model_provider=megatron_actor_model_provider, use_distributed_optimizer=True)
    # Fetch the model at current rank
    actor_module = hybrid_engine.this_rank_models
    actor_modules_list = []
    if isinstance(actor_module, torch.nn.ModuleList):
        for module in actor_module:
            actor_modules_list.append(module)
    actor_module = actor_modules_list
    print(f'actor_module: {len(actor_module)}')
    # important
    config = {config.model.path}

    load_megatron_gptmodel_weights(config,
                                   actor_model_config,
                                   actor_module,
                                   params_dtype=torch.bfloat16,
                                   is_value_model=False)
    log_gpu_memory_usage('After AllGatherPPModel init')
    if torch.distributed.get_rank() == 0:
        print_model_size(actor_module[0])

    hybrid_engine.load_params_to_cuda()
    # broadcast the parameters from pp rank to other ranks
    hybrid_engine.allgather_params()
    # obtain name to parameters in pp/vpp
    params = hybrid_engine.get_all_params()
    # update the param name for the
    params = normalize_pp_vpp_params(params=params,
                                     num_hidden_layers=actor_model_config.num_hidden_layers,
                                     layer_name='layers')

    rollout = vLLMRollout(actor_module=params,
                          config=self.config.rollout,
                          tokenizer=self.tokenizer,
                          model_hf_config=self.actor_model_config,
                          train_tp=mpu.get_tensor_model_parallel_world_size())
    log_gpu_memory_usage('After building vllm rollout')

    # perform weight resharding between actor and rollout
    sharding_manager = MegatronVLLMShardingManager(module=self.hybrid_engine,
                                                   inference_engine=rollout.inference_engine,
                                                   model_config=self.actor_model_config,
                                                   layer_name_mapping=layer_name_mapping)
    log_gpu_memory_usage('After building sharding manager')
