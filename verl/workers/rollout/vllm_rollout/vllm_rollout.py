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

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

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

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
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
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        
        self.tokenizer = tokenizer

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
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
            
        print(f"prompts: {len(prompts)}, {prompts}")

        input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = input_ids.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, input_ids[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        print("Initial vllm input size: ", [len(init_ids) for init_ids in idx_list])
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)
        
        # string-format conversation to debug
        batch_chat = [[] for _ in range(batch_size * self.config.n)]
        
        # lurui: feature for multi-turn ai-search
        if self.config.get('multi_turn', False):
            # 提前把 input_ids 给重复好
            if self.config.n > 1 and do_sample:
                input_ids = input_ids.repeat_interleave(self.config.n, dim=0)
                # attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
                position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
                batch_size = batch_size * self.config.n

            response = output[0].to(input_ids.device)
            log_probs = output[1].to(input_ids.device)
                
            # with open("/workspace/lurui-yun/deep_research/verl/logs/idx_list.txt", "w") as f:
            #     print(idx_list, file=f)
            #     print("type(idx_list)", file=f)
            #     print(type(idx_list), file=f)
            
            import requests
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def call_observation_api(text: str) -> List[str]:
                url = "http://172.18.80.255:8888/observation_kilt/"
                payload = {"content": text}
                try:
                    api_response = requests.post(url, json=payload)
                    # return api_response.json()[0]['content']
                    return api_response.json()
                except Exception as e:
                    print(f"API call failed: {e}")
                    return None

            # Initialize tensors for final sequences
            # TODO: can set config parameter: max_turns
            max_turns = 5
            batch_size = len(response)
            max_seq_length = self.config.response_length + input_ids.size(1)
            
            final_sequence = torch.full(
                (batch_size, max_seq_length), 
                self.pad_token_id, 
                dtype=input_ids.dtype, 
                device=input_ids.device
            )
            final_attention_mask = torch.zeros(
                (batch_size, max_seq_length), 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )
            final_loss_mask = torch.zeros(
                (batch_size, max_seq_length), 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )

            for i, resp in enumerate(response):
                current_length = input_ids[i].size(0)
                final_sequence[i, :current_length] = input_ids[i]
                final_sequence[i, current_length:current_length+resp.size(0)] = resp
                final_attention_mask[i, :current_length] = (input_ids[i] != self.pad_token_id).long()
                final_attention_mask[i, current_length:current_length+resp.size(0)] = 1
                final_loss_mask[i, current_length:current_length+resp.size(0)] = 1

            # only for glm
            # observation_id = self.tokenizer.encode("<|observation|>")[-1]
            # user_id = self.tokenizer.encode("<|user|>")[-1]
            # assert observation_id == 151338 and user_id == 151336, f"observation_id: {observation_id}, user_id: {user_id}, expected: 151338, 151336"
            
            with open("/workspace/lurui-yun/deep_research/verl/logs/final_sequence_init.json", "w") as f:
                import json
                json.dump(final_sequence.tolist(), f)
            
            for i in range(batch_size):
                batch_chat[i].append({
                    "role": "user",
                    "content": self.tokenizer.decode(final_sequence[i], skip_special_tokens=False).replace("<|endoftext|>", "").strip().split("<|assistant|>")[0]
                })
                batch_chat[i].append({
                    "role": "assistant",
                    "content": self.tokenizer.decode(response[i], skip_special_tokens=False).replace("<|endoftext|>", "").strip()
                })
            
            # Process each response in batch
            for turn in range(max_turns):
                # Decode all responses in batch
                decoded_responses = []
                observation_indices = []

                # Gather decoded responses and observation indices
                for check_idx, resp in enumerate(final_sequence):
                    resp_trim = resp[resp != self.pad_token_id]

                    stop_token_id = resp_trim[-1]
                    stop_token = self.tokenizer.decode([stop_token_id])
                    print(f"Response #{check_idx}, stop token: {stop_token_id} - {stop_token}")
                    
                    # only finish with <|observation|> can be multi-turn
                    # for glm judge
                    # if stop_token_id == observation_id:
                    if stop_token == '<|observation|>':
                        text = self.tokenizer.decode(resp_trim, skip_special_tokens=False)
                        text = text.split("<|assistant|>")[-1].split("<|observation|>")[0].strip()
                        decoded_responses.append(text)
                        observation_indices.append(check_idx)
                    
                    # for qwen judge
                    # response_str = self.tokenizer.decode(resp_trim)
                    # if stop_token == '<|observation|>':
                    #     # if turn == 0:
                    #     #     print("Observation response found!")
                    #     #     print("Response: ", response_str)
                    #     text = self.tokenizer.decode(resp_trim, skip_special_tokens=False)
                    #     # for qwen
                    #     text = text.split("<|assistant|>")[-1].split("<|observation|>")[0].strip()
                    #     decoded_responses.append(text)
                    #     observation_indices.append(check_idx)
                
                with open("/workspace/lurui-yun/deep_research/verl/logs/decoded_responses.json", "w") as f:
                    import json
                    json.dump(decoded_responses, f)

                # Exit the loop if no observation responses are found
                if not observation_indices:
                    break

                observations = [None] * batch_size
                
                # may be heavy parallel, depends on API
                # with ThreadPoolExecutor(max_workers=10) as executor:
                with ThreadPoolExecutor(max_workers=len(observation_indices)) as executor:
                    future_to_idx = {executor.submit(call_observation_api, text): idx 
                                    for idx, text in zip(observation_indices, decoded_responses)}
                    
                    for future in as_completed(future_to_idx):
                        idx_pos = future_to_idx[future]
                        try:
                            result = future.result()
                            observations[idx_pos] = result
                        except Exception as e:
                            print(f"Error processing response {idx_pos}: {e}")
                
                print(f"len(observation_indices): {len(observation_indices)}")
                print(f"observation_indices: {observation_indices}")
                
                # observation can be None 
                observation_indices = [i for i, obv in enumerate(observations) if obv]

                # sequences with added observations to inference
                current_sequences = []
                
                # with open("/workspace/lurui-yun/deep_research/verl/logs/observation.json", "w") as f:
                #     import json
                #     json.dump(observations, f)

                def remove_trailing_pad_tokens(tensor, pad_token_id):
                    tensor = tensor.flip(dims=[0])
                    non_pad_indices = (tensor != pad_token_id).nonzero(as_tuple=True)[0]
                    if len(non_pad_indices) > 0:
                        first_non_pad_index = non_pad_indices[0].item()
                        tensor = tensor[first_non_pad_index:]
                    tensor = tensor.flip(dims=[0])
                    return tensor
                
                exceed_indices = []
                for i in observation_indices:
                    current_sequence = remove_trailing_pad_tokens(final_sequence[i], self.pad_token_id)

                    assert observations[i] != None, f"observations[{i}] is None"
                    
                    obv_combined = [obv['metadata'] + '\n'+ obv['content'] for obv in observations[i]]
                    
                    # for glm observation context
                    obs_text = f"{'<|observation|>'.join(obv_combined)}<|assistant|>\n"
                    obs_ids = self.tokenizer(obs_text, add_special_tokens=True, return_tensors="pt")["input_ids"][:, 2:].to(input_ids.device)
                    
                    # for (current) qwen observation context
                    # connect_obv = '\n<|im_start|>\n' + '<|im_end|>\n<|im_start|>observation\n'.join(obv_combined)
                    # obs_text = connect_obv + "<|im_end|>\n<|im_start|>assistant\n"
                    # obs_text = f"{'<|observation|>'.join(obv_combined)}<|assistant|>\n"
                    # obs_ids = self.tokenizer(obs_text, add_special_tokens=True, return_tensors="pt")["input_ids"][:, :].to(input_ids.device)
                    
                    batch_chat[i].append({
                        "role": "observation",
                        "content": obs_text
                    })
                    
                    # if current_length & observation length > max_length
                    # do not inference for this observation
                    exceed_length = False
                    pad_pos = current_sequence.size(0)
                    max_length = final_sequence.size(1)
                    if pad_pos + obs_ids.size(1) > max_length:
                        obs_ids = obs_ids[:, :max_length - pad_pos]
                        # observation_indices.remove(i)
                        exceed_indices.append(i)
                        exceed_length = True
                    
                    # for observation, attention_mask must set to 1, loss_mask set to 0
                    final_sequence[i, pad_pos:pad_pos + obs_ids.size(1)] = obs_ids.squeeze(0)
                    final_attention_mask[i, pad_pos:pad_pos + obs_ids.size(1)] = 1
                    current_sequence = torch.cat([current_sequence.unsqueeze(0), obs_ids], dim=1).squeeze(0)
                    current_sequence = current_sequence[current_sequence != self.pad_token_id]
                    
                    if not exceed_length:
                        current_sequences.append(current_sequence)
                
                # remove exceed observation
                observation_indices = [i for i in observation_indices if i not in exceed_indices]
                
                # with open(f"/workspace/lurui-yun/deep_research/verl/logs/final_sequence_add_observation_{turn}.json", "w") as f:
                #     import json
                #     json.dump(final_sequence.tolist(), f)

                current_sequences = [seq.squeeze().tolist() for seq in current_sequences]
                
                # if turn == 0:
                #     with open("/workspace/lurui-yun/deep_research/verl/logs/current_sequences.json", "w") as f:
                #         import json
                #         json.dump(current_sequences, f)
                
                if not current_sequences:
                    break
                
                print(f"New rollout at turn {turn} input size: ", [len(curr_seq) for curr_seq in current_sequences])
                print("New self.sampling_params: ", self.sampling_params)
                
                # fix by lurui: for new batch (max batch_size * n), each is different, set n = 1
                kwargs_for_multi_turn = {
                    'n': 1
                }
                
                # Generate next responses
                with self.update_sampling_params(**kwargs_for_multi_turn):
                    next_outputs = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=current_sequences,
                        use_tqdm=False
                    )
                    next_outputs = next_outputs[0].to(input_ids.device)
                
                assert len(current_sequences) == next_outputs.size(0), f"len(current_sequences): {len(current_sequences)}, next_outputs.size(0): {next_outputs.size(0)}"
                
                assert len(observation_indices) == next_outputs.size(0), f"len(observation_indices): {len(observation_indices)}, next_outputs.size(0): {next_outputs.size(0)}"
                
                # with open("/workspace/lurui-yun/deep_research/verl/logs/next_outputs.json", "w") as f:
                #     import json
                #     json.dump(next_outputs.tolist(), f)

                # Update responses and final tensors
                for idx, i in enumerate(observation_indices):
                    next_output = next_outputs[idx]
                    next_output_trim = remove_trailing_pad_tokens(next_output, self.pad_token_id)
                    
                    batch_chat[i].append({
                        "role": "assistant",
                        "content": self.tokenizer.decode(next_output_trim, skip_special_tokens=False).replace("<|endoftext|> ", "")
                    })
                    
                    # Find the first pad_token_id position in final_sequence
                    pad_pos = remove_trailing_pad_tokens(final_sequence[i], self.pad_token_id).size(0)
                    
                    # Update final tensors
                    seq_length = pad_pos + next_output_trim.size(0)
                    if seq_length > final_sequence.size(1):
                        seq_length = final_sequence.size(1)
                        next_output_trim = next_output_trim[:seq_length - pad_pos]
                    final_sequence[i, pad_pos:seq_length] = next_output_trim
                    
                    # for answer by model, attention_mask and loss_mask all set to 1
                    final_attention_mask[i, pad_pos:seq_length] = 1
                    final_loss_mask[i, pad_pos:seq_length] = 1
                
                # with open(f"/workspace/lurui-yun/deep_research/verl/logs/final_sequence_{turn}.json", "w") as f:
                #     import json
                #     json.dump(final_sequence.tolist(), f)
            
                print(f"Turn #{turn} done!")
            
            # real-response(remove input_ids)
            response = final_sequence[:, input_ids.size(1):]
            
            trimmed_responses = []
            for resp in response:
                trimmed_resp = resp[resp != self.pad_token_id]
                trimmed_responses.append(trimmed_resp)
            
            # Find the maximum length of the trimmed responses
            max_length = max(len(trimmed_resp) for trimmed_resp in trimmed_responses)
            
            # Pad the trimmed responses to the maximum length
            padded_responses = torch.full((len(trimmed_responses), max_length), self.pad_token_id, dtype=response.dtype)
            for i, trimmed_resp in enumerate(trimmed_responses):
                padded_responses[i, :len(trimmed_resp)] = trimmed_resp
            
            response = padded_responses.to(input_ids.device)
            
            # with open("/workspace/lurui-yun/deep_research/verl/logs/response_out_of_loop.json", "w") as f:
            #     import json
            #     json.dump(response.tolist(), f)
                
            # with open("/workspace/lurui-yun/deep_research/verl/logs/batch_chat.json", "w") as f:
            #     import json
            #     json.dump(batch_chat, f)
            
            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)
            
            attention_mask = final_attention_mask[:, :input_ids.size(1) + self.config.response_length].to(input_ids.device)
            
            loss_mask = final_loss_mask[:, :input_ids.size(1) + self.config.response_length].to(input_ids.device)
            
            # with open("/workspace/lurui-yun/deep_research/verl/logs/attention_mask.json", "w") as f:
            #     import json
            #     json.dump(attention_mask.tolist(), f)
            
            seq = torch.cat([input_ids, response], dim=-1)

            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

            response_position_ids = position_ids[:, -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            
            # attention_mask is already prepared before
            # response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            # attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        else:
            # single-turn: original code
            response = output[0].to(input_ids.device)
            log_probs = output[1].to(input_ids.device)

            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

            if self.config.n > 1 and do_sample:
                input_ids = input_ids.repeat_interleave(self.config.n, dim=0)
                attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
                position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
                batch_size = batch_size * self.config.n
            
            seq = torch.cat([input_ids, response], dim=-1)
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
            loss_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        print(f"input_ids: {input_ids.shape}")
        print(f"response: {response.shape}")
        print(f"seq: {seq.shape}")
        print(f"attention_mask: {attention_mask.shape}")
        print(f"loss_mask: {loss_mask.shape}")
        print(f"position_ids: {position_ids.shape}")
        
        observations_times = torch.tensor([sum([1 for turn in chat if turn['role'] == 'observation']) for chat in batch_chat]).to(input_ids.device)
        
        print(f"observations_times: {observations_times}")  

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': input_ids,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
                'observations_times': observations_times
            },
            batch_size=batch_size)

        import json
        
        # for single turn
        # with open('/workspace/lurui-yun/deep_research/verl/logs/generate_sequences_call.json', 'w') as f:
        #     f.write(json.dumps({
        #         'prompts': input_ids.tolist(),
        #         'responses': response.tolist(),
        #         'input_ids': seq.tolist(),
        #         'attention_mask': attention_mask.tolist(),
        #         'position_ids': position_ids.tolist(),
        #     }))
        #     f.write("\n")
        
        # for multi-turn
        with open('/workspace/lurui-yun/deep_research/verl/logs/generate_sequences_call.json', 'w') as f:
            f.write(json.dumps({
                'prompts': input_ids.tolist(),
                'responses': response.tolist(),
                'input_ids': seq.tolist(),
                'attention_mask': attention_mask.tolist(),
                'loss_mask': loss_mask.tolist(),
                'position_ids': position_ids.tolist(),
                'decoded_responses': decoded_responses,
                'observations': observations,
                'batch_chat': batch_chat
            }))
            f.write("\n")

        # 用比较好看的方式打印 batch_chat
        # for i in range(len(batch_chat)):
        #     print(f"Conversation {i}:")
        #     for turn in batch_chat[i]:
        #         print(f"***{turn['role']}***")
        #         print(f"{turn['content']}")
        #         print("*" * 20)
        #     print("\n")


        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
