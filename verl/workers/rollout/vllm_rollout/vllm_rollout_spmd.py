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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
import time
import asyncio

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            swap_space=80,
            # quantization="fp8",
            # cpu_offload_gb=40,
            kv_cache_dtype=config.kv_cache_dtype,
            calculate_kv_scales=(config.kv_cache_dtype == "fp8"),
            enable_prefix_caching=False,
            preemption_mode="swap",
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            temperature=1.,
            max_tokens=config.response_length,
            # repetition_penalty=1.,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
        else:
            self.inference_engine.wake_up()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 0.95,
                # 'top_k': -1,
                # 'min_p': 0.0,
                'temperature': 0.6,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)

        response = pad_2d_list_to_length(response, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        else:
            time_now = time.time()
            # print("waiting for engine to sleep {}".format(time_now))
            self.inference_engine.sleep(level=1)
            # print("engine is sleeping {}".format(time_now))

        return DataProto(batch=batch)


class vLLMRolloutWithSelf(vLLMRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer

        query_start = "<|im_start|>user<|im_sep|>"
        query_end = "<|im_end|><|im_start|>assistant<|im_sep|>"

        tool_start = "<tool>"
        tool_end = "</tool>"
        tool_response_start = "<tool_response>"
        tool_response_end = "</tool_response>"
        think_start = "<think>"
        think_end = "</think>"

        self.query_start_id_list = tokenizer.encode(query_start, add_special_tokens=False)
        self.query_end_id_list = tokenizer.encode(query_end, add_special_tokens=False)

        self.tool_start_id_list = tokenizer.encode(tool_start, add_special_tokens=False)
        self.tool_end_id_list = tokenizer.encode(tool_end, add_special_tokens=False)

        self.tool_response_start_id_list = tokenizer.encode(tool_response_start, add_special_tokens=False)
        self.tool_response_end_id_list = tokenizer.encode(tool_response_end, add_special_tokens=False)

        self.think_start_id_list = tokenizer.encode(think_start, add_special_tokens=False)
        self.think_end_id_list = tokenizer.encode(think_end, add_special_tokens=False)

    def extract_tool_content_from_string(self, text: str, tag: str = "tool") -> str:
        try:
            start_tag = '<'+ tag + '>'
            end_tag = '</'+ tag + '>'
            assert text.strip().endswith(end_tag)
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag):end_pos].strip()
        except ValueError:
            # print(f"extract_search_content failed: {text}")
            return ""

    def find_ids_in_list(self, list, target_ids):
        # convert to tuple for faster comparison
        list_tup = tuple(list)
        target_ids_tup = tuple(target_ids)

        # find the last target_ids in the list
        start_pos = -1
        for i in range(len(list_tup)-len(target_ids_tup)):
            if list_tup[i:i+len(target_ids_tup)] == target_ids_tup:
                start_pos = i
                break
        
        return start_pos

    def extract_tool_content_from_ids(self, list):
        start_ids = self.tool_response_start_id_list
        end_ids = self.tool_response_end_id_list

        # assert that the list ends with the end_ids list
        assert list[-len(end_ids):] == end_ids

        start_pos = self.find_ids_in_list(list, start_ids)

        if start_pos == -1:
            return None
        
        return list[start_pos + len(start_ids): - len(end_ids)]


    def get_result_mask(self, response_id: torch.Tensor, result_start_token: int, result_end_token: int, dtype=torch.int64) -> torch.Tensor:
        batch_size, seq_len = response_id.shape
        mask = torch.ones_like(response_id, dtype=dtype)
        
        # 找到所有<result>和</result>的位置
        start_positions = (response_id == result_start_token).nonzero()
        end_positions = (response_id == result_end_token).nonzero()
        
        # 对每个batch处理
        for i in range(batch_size):
            batch_starts = start_positions[start_positions[:, 0] == i, 1]
            batch_ends = end_positions[end_positions[:, 0] == i, 1]
            
            # 确保start和end数量相等
            min_pairs = min(len(batch_starts), len(batch_ends))
            assert len(batch_starts) - len(batch_ends) <= 1
            if len(batch_starts) > len(batch_ends):
                batch_ends = torch.cat((batch_ends, torch.tensor([seq_len - 1], device=response_id.device)))
            for j in range(len(batch_starts)):
                start_idx = batch_starts[j]
                end_idx = batch_ends[j]
                if start_idx < end_idx:
                    # 将<result>和</result>之间的内容mask为0
                    mask[i, start_idx:end_idx+1] = 0
        
        return mask
    
    def remove_tagged_block(self, response, end_ids) -> torch.Tensor:

        end_position = self.find_ids_in_list(response, end_ids)

        if end_position == -1:
            return response
        
        return response[end_position+len(end_ids):]


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
        else:
            self.inference_engine.wake_up()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                # 'top_k': -1,
                # 'min_p': 0.0,
                'temperature': 0.6,
                'n': 1  # if greedy, only 1 response
            }

        with self.update_sampling_params(**kwargs):
            from tqdm import tqdm
            outer_iterator = tqdm(idx_list) if not do_sample else idx_list
            
            result_mask_list = []
            output_ids_list = []

            async def compute_result(input_ids):
                curr_max_tokens = self.sampling_params.max_tokens
                curr_input_ids = input_ids.copy()
                curr_result_mask = []
                while curr_max_tokens > 0:
                    # with self.update_sampling_params(n=1, stop_token_ids=[self.tool_end_id_list[-1]], max_tokens=curr_max_tokens):
                    #     output = self.inference_engine.generate(
                    #         prompts=None,  # because we have already convert it to prompt token id
                    #         sampling_params=self.sampling_params,
                    #         prompt_token_ids=curr_input_ids,
                    #         use_tqdm=False)
                    with self.update_sampling_params(n=1, stop=['</tool>'], max_tokens=curr_max_tokens, detokenize=True, include_stop_str_in_output = True):
                        output = self.inference_engine.generate(
                            prompts=None,  # because we have already convert it to prompt token id
                            sampling_params=self.sampling_params,
                            prompt_token_ids=curr_input_ids,
                            use_tqdm=False)

                    curr_output_ids = output[0].outputs[0].token_ids

                    # if last token is eos
                    if curr_output_ids[-1] == self.tokenizer.eos_token_id:
                        curr_input_ids += curr_output_ids
                        curr_max_tokens -= len(curr_output_ids)
                        curr_result_mask.extend([1] * len(curr_output_ids))
                        break

                    # if the output ends with self.tool_end_id_list, we need to extract the tool content and do the tool query
                    elif curr_output_ids[-len(self.tool_end_id_list):] == self.tool_end_id_list:
                        tool_content = self.extract_tool_content_from_ids(curr_output_ids, self.tool_start_id_list, self.tool_end_id_list)
                        tool_query = self.query_start_id_list + tool_content + self.query_end_id_list
                        tool_max_tokens = self.sampling_params.max_tokens - len(tool_query)
                        with self.update_sampling_params(n=1, max_tokens=tool_max_tokens):
                            tool_output = self.inference_engine.generate(
                                prompts=None,  # because we have already convert it to prompt token id
                                sampling_params=self.sampling_params,
                                prompt_token_ids=tool_query,
                                use_tqdm=False)
                        tool_output_ids = tool_output[0].outputs[0].token_ids
                        tool_output_ids = tool_output_ids[len(tool_query):]
                        tool_output_ids = self.remove_tagged_block(tool_output_ids, self.think_end_id_list)
                        if tool_output_ids[-1] == self.tokenizer.eos_token_id:
                            tool_output_ids = tool_output_ids[:-1]

                        curr_result_mask.extend([1] * len(curr_output_ids))
                        prefix_len = len(curr_output_ids)
                        # update curr_output_ids with search result
                        curr_output_ids = curr_output_ids + self.tool_response_start_id_list + tool_output_ids + self.tool_response_end_id_list
                        curr_result_mask.extend([0] * (len(curr_output_ids) - prefix_len))

                        curr_output_ids = curr_output_ids[:curr_max_tokens]
                        curr_result_mask = curr_result_mask[:curr_max_tokens]
                        
                        # prepare for continue thinking
                        curr_input_ids += curr_output_ids
                        curr_max_tokens -= len(curr_output_ids)
                    
                    # if the output ends with other tags, we just continue
                    else:
                        curr_input_ids += curr_output_ids
                        curr_max_tokens -= len(curr_output_ids)
                        curr_result_mask.extend([1] * len(curr_output_ids))

                # output_ids_list.append(curr_input_ids[len(input_ids):])
                # result_mask_list.append(curr_result_mask)
                return curr_input_ids[len(input_ids):], curr_result_mask

            tasks = []
            for input_ids in outer_iterator:
                for _ in range(self.sampling_params.n):
                    tasks.append(compute_result(input_ids))
            
            # run tasks concurrently and collect results
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(asyncio.gather(*tasks))
            for output_ids, result_mask in results:
                output_ids_list.append(output_ids)
                result_mask_list.append(result_mask)

        response = pad_2d_list_to_length(output_ids_list, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # result_mask = torch.cat((torch.zeros_like(attention_mask), result_mask), dim=-1)
        
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                # 'result_mask': result_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        else:
            # time_now = time.time()
            # print("waiting for engine to sleep {}".format(time_now))
            self.inference_engine.sleep(level=1)
            # print("engine is sleeping {}".format(time_now))

        return DataProto(batch=batch)