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

os.environ['NCCL_DEBUG'] = 'WARN'

import torch
import torch.distributed

from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy
from verl.utils.torch_functional import logprobs_from_logits, entropy_from_logits

from verl.utils.debug import log_gpu_memory_usage

from megatron.core import mpu


class Utils:
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get('LOCAL_RANK', '0'))

    @staticmethod
    def initialize_distributed():
        print(f'Initializing torch.distributed with rank: {Utils.rank}, world_size: {Utils.world_size}')
        torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '7000')
        init_method += master_ip + ':' + master_port
        torch.distributed.init_process_group(backend='nccl',
                                             world_size=Utils.world_size,
                                             rank=Utils.rank,
                                             init_method=init_method)
        print(f'successfully created process group')

    @staticmethod
    def destroy_model_parallel():
        mpu.destroy_model_parallel()
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    @staticmethod
    def initialize_model_parallel(tensor_model_parallel_size=1,
                                  pipeline_model_parallel_size=1,
                                  virtual_pipeline_model_parallel_size=None,
                                  pipeline_model_parallel_split_rank=None):
        mpu.destroy_model_parallel()
        if not torch.distributed.is_initialized():
            Utils.initialize_distributed()
        mpu.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size,
                                      virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank)


def test_vocab_parallel_entropy():
    # check vocab_parallel_entropy
    Utils.world_size = 8
    Utils.initialize_model_parallel(8, 1)

    batch_size = 2
    seqlen = 128
    vocab_size = 155136

    logits = torch.randn(batch_size * seqlen, vocab_size, device='cuda', requires_grad=True)
    target = torch.randint(low=0, high=vocab_size, size=(batch_size * seqlen,), device='cuda', dtype=torch.int64)

    # broadcast across tp
    torch.distributed.broadcast(logits,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    torch.distributed.broadcast(target,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())

    tp_rank = mpu.get_tensor_model_parallel_rank()
    vocab_size_per_tp = vocab_size // mpu.get_tensor_model_parallel_world_size()

    # get the local logits of each tp
    vocab_parallel_logits = logits.clone().detach()[:, tp_rank * vocab_size_per_tp:(tp_rank + 1) *
                                                    vocab_size_per_tp].requires_grad_()
    logits.grad = None
    vocab_parallel_logits.grad = None

    log_gpu_memory_usage('begin')
    output_entropy = vocab_parallel_entropy(vocab_parallel_logits)
    log_gpu_memory_usage('after forward')
    grad_output = torch.randn_like(output_entropy)
    output_entropy.backward(grad_output)
    log_gpu_memory_usage('after backward')

    target_entropy = entropy_from_logits(logits)
    torch.testing.assert_close(output_entropy, target_entropy)
    target_entropy.backward(grad_output)
    torch.testing.assert_close(logits.grad[:, tp_rank * vocab_size_per_tp:(tp_rank + 1) * vocab_size_per_tp],
                               vocab_parallel_logits.grad)
    # make sure logits is not altered
    torch.testing.assert_close(logits[:, tp_rank * vocab_size_per_tp:(tp_rank + 1) * vocab_size_per_tp],
                               vocab_parallel_logits)

    if mpu.get_tensor_model_parallel_rank() == 0:
        print('test_vocab_parallel_entropy passes')

    Utils.destroy_model_parallel()


if __name__ == '__main__':
    test_vocab_parallel_entropy()
