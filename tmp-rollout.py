# input_ids during generation should not contain padding
# TODO dynamic batching
# receive signal when container exited unexpectedly
# TODO: unstable docker exclusion
# padding: [lpad] | [input] | [response] | [rpad]

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
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import numpy as np
import requests
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl.utils.swedev_utils import *
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams
from vllm.logger import init_logger

logger = init_logger(__name__)

SWE_DEBUG = True

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

class MultiTurnInference:
    def __init__(self, model, tokenizer, inference_engine, max_turns=30):
        self.model = model
        self.tokenizer = tokenizer
        self.inference_engine = inference_engine
        self.max_turns = max_turns
        self.pad_token_id = tokenizer.pad_token_id
        
    def _initialize_runtime(self, instance_id):
        url = get_api(type="start")
        payload = {"instance_hash": instance_id}
        try:
            api_response = requests.post(url, json=payload)
            return api_response.json()
        except Exception as e:
            print(f"Initializing - API call failed: {e}")
            return None

    def _call_observation_api(self, sid, text: str) -> List[str]:
        """Call external API to get observation for a given text"""
        if type(sid) == torch.Tensor:
            sid = sid.item()
        url = get_api(type="action")
        payload = {
            "sid": sid,
            "content": text,
        }
        try:
            api_response = requests.post(url, json=payload)
            return api_response.json()
        except Exception as e:
            print(f"Observation - API call failed: {e}")
            return None

    def _call_postprocess_api(self, sid: str):
        url = get_api(type="postprocess")
        if type(sid) == torch.Tensor:
            sid = sid.item()        
        payload = {"sid": sid}
        try:
            api_response = requests.post(url, json=payload)
            return api_response.json()
        except Exception as e:
            print(f"Postprocess - API call failed: {e}")
            return None

    def _remove_padding(self, tensor: torch.Tensor) -> torch.Tensor:
        """Remove padding tokens from the end of a tensor"""
        tensor = tensor.flip(dims=[0])
        non_pad_indices = (tensor != self.pad_token_id).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) > 0:
            first_non_pad_index = non_pad_indices[0].item()
            tensor = tensor[first_non_pad_index:]
        return tensor.flip(dims=[0])

    def _is_stop(self, response):
        is_finish = "<function=finish>" in response
        return is_finish
    
    def generate_batch_response(self, prompt_token_ids_list: List[int], sampling_params: dict) -> List:
        """Generate responses for a batch of prompts"""
        try:
            outputs = self.inference_engine.generate(
                prompts=None,
                sampling_params=sampling_params,
                prompt_token_ids=prompt_token_ids_list,
                use_tqdm=False
            )
            return outputs[0]
        except Exception as e:
            print(f'batch_response Error: {e}')
            traceback.print_exc()
            return None

    def process_batch_prompts(self, sampling_params, input_ids_list, instance_ids, max_length) -> List[dict]:
        """Process a batch of prompts through multiple turns of conversation (serial approach with parallel API calls)"""
        try:
            batch_size = len(instance_ids)
            token_histories = [] # list of token histories for each instance (0, 1 for system prompt and initial instruction)
            response_ranges_list = []
            sids = []
            completed = [False] * batch_size
            errors = [None] * batch_size
            results = [None] * batch_size
            input_length = [len(input_ids) for input_ids in input_ids_list]

            # for i, instance_id in enumerate(instance_ids):
            #     init_result = self._initialize_runtime(instance_id.item())
            #     print(init_result)
            #     sid = init_result['sid']
            #     sids.append(sid)
            #     token_histories.append(input_ids_list[i].copy())
            #     response_ranges_list.append([])

            def _initialize_runtime(idx, instance_id):
                result = self._initialize_runtime(instance_id)
                print('init result', result)
                return idx, result["sid"]

            with ThreadPoolExecutor(max_workers=min(len(instance_ids), 10)) as executor:
                future_to_idx = {executor.submit(_initialize_runtime, idx, instance_id.item()): (idx, instance_id) for idx, instance_id in enumerate(instance_ids)}
                for future in as_completed(future_to_idx):
                    try:
                        result_idx, sid = future.result()
                        print(f"Got SID: {result_idx}, {sid}, SIDS: {sids}")
                        token_histories.append(input_ids_list[result_idx].copy())
                        sids[result_idx] = sid
                        response_ranges_list.append([])
                    except Exception as e:
                        print(f"Error processing instance: {e}")
                        traceback.print_exc()

            # Main generation loop
            for turn in range(self.max_turns):
                if all(completed):
                    break

                active_indices = [i for i, is_completed in enumerate(completed) if not is_completed]
                active_token_ids = [token_histories[i] for i in active_indices]
                if not active_token_ids:
                    break

                outputs = self.generate_batch_response(active_token_ids, sampling_params)

                def _generate_traj(idx, output_idx):
                    try:
                        # here should not contain any right pad
                        output_token_ids = self._remove_padding(outputs[idx]).tolist()
                        response_start_idx = len(token_histories[output_idx]) - input_length[output_idx]
                        response_length = len(output_token_ids)
                        response_range = [response_start_idx, response_start_idx + response_length]
                        # haoran: checking from tokens to be more efficient?
                        response = self.tokenizer.decode(output_token_ids, skip_special_tokens=False)
                        
                        new_token_history = token_histories[output_idx] + output_token_ids + get_generation_tokens(role="user") # for observation of the current turn
                        
                        # Check if LLM generation triggers stop condition
                        if self._is_stop(response):
                            return {
                                "output_idx": output_idx,
                                "token_history": new_token_history[:-len(get_generation_tokens(role="user"))],
                                "response_range": response_range,
                                "is_completed": True,
                                "error": None
                            }

                        # Call API to get observation
                        obs = self._call_observation_api(sids[output_idx], response)

                        # Prepare result
                        result = {
                            "output_idx": output_idx,
                            "token_history": new_token_history,
                            "response_range": response_range,
                            "is_completed": False,
                            "error": None
                        }
                        
                        # TODO: retry logic
                        # if "error" in obs and obs["error"]:
                        #     result["token_history"] = new_token_history[:-len(get_generation_tokens(role="user"))]
                        #     result["error"] = obs["error"]
                        #     result["is_completed"] = True
                        #     return result

                        obs_tokens = self.tokenizer(obs["content"], add_special_tokens=False)["input_ids"]

                        if len(new_token_history + obs_tokens) > max_length:
                            result["token_history"] = new_token_history[:-len(get_generation_tokens(role="user"))]
                            result["error"] = OBS_ERROR_MSG
                            result["is_completed"] = True
                            return result

                        result["token_history"] = new_token_history + obs_tokens + get_generation_tokens(role="assistant")
                        return result
                    except Exception as e:
                        print(f'Error when generating trajectory: {e}')
                        return {
                            "output_idx": output_idx,
                            "is_completed": True,
                            "error": e
                        }
                    
                with ThreadPoolExecutor(max_workers=min(len(active_indices), 10)) as executor:
                    future_to_idx = {
                        executor.submit(_generate_traj, idx, output_idx): (idx, output_idx)
                        for idx, output_idx in enumerate(active_indices)
                    }

                    for future in as_completed(future_to_idx):
                        idx, output_idx = future_to_idx[future]
                        try:
                            result = future.result()
                            # print(f"Task {idx} completed. Output Index: {output_idx}, Result: {result}")
                            if "error" in result.keys() and result["error"] and result["is_completed"]: # Error when rollouting
                                errors[output_idx] = f"processing_error: {str(e)}"
                                completed[output_idx] = True
                            else:
                                token_histories[output_idx] = result["token_history"]
                                response_ranges_list[output_idx].append(result["response_range"])
                                completed[output_idx] = result["is_completed"]
                                errors[output_idx] = result["error"]
                        except Exception as e:
                            print(f"Error processing instance {output_idx}: {e}")
                            traceback.print_exc()
                            errors[output_idx] = f"processing_error: {str(e)}"
                            completed[output_idx] = True
                                            
            def finish_single_instance(index):
                response_tokens = None
                if len(response_ranges_list[index]) > 0:
                    response_tokens = token_histories[index][input_length[index]:]
                else:
                    response_tokens = torch.tensor([], dtype=torch.long)

                # Remove remote runtime in shouyun machine
                self._call_postprocess_api(sid)

                return {
                    "response_ids": token_histories[index][input_length[index]:],
                    "response": response_tokens,
                    "response_ranges": response_ranges_list[index],
                    "error": errors[index],
                    "sid": sids[index]
                }

            with ThreadPoolExecutor(max_workers=10) as executor: 
                futures = {executor.submit(finish_single_instance, i): i for i in range(batch_size)}
                results = [None] * batch_size
                failed_indices = []

                for future in as_completed(futures):
                    index = futures[future] 
                    try:
                        result = future.result()
                        results[index] = result
                    except Exception as e:
                        print(f"Error processing instance {index}: {e}")
                        traceback.print_exc()
                        failed_indices.append(index)

            if failed_indices:
                print(f"The following tasks failed: {failed_indices}")
                
            # Prepare final results
            # for i in range(batch_size):            
            #     response_tokens = None
            #     if len(response_ranges_list[i]) > 0:
            #         response_tokens = token_histories[i][input_length[i]:]
            #     else:
            #         response_tokens = torch.tensor([], dtype=torch.long)

            #     # remove remote runtime in shouyun machine
            #     self._call_close_runtime_api(sids[i])

            #     results[i] = {
            #         "response_ids": token_histories[i][input_length[i]:],
            #         "response": response_tokens,
            #         "response_ranges": response_ranges_list[i],
            #         "error": errors[i],
            #         "sid": sids[i]
            #     }

            return results

        except Exception as e:
            print(f"Error when process_batch_prompts: {e}")
            traceback.print_exc()
            
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

        max_turns = self.config.get('max_turns', 30)
        kwargs = dict(
            n=1,
            logprobs=1,
            max_tokens=config.response_length-max_turns*4,
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

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()
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
        # print(f'Prompt batch size at rollout generator: {len(prompts)}')
        assert self.config.get('multi_turn', False), "multi_turn must be enabled"
        do_sample = prompts.meta_info.get('do_sample', True)
        assert do_sample, "do_sample must be enabled"
        sequences = []
        input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        input_ids = input_ids.repeat_interleave(self.config.n, dim=0)
        input_length = input_ids.size(1)
        attention_mask = prompts.batch['attention_mask']
        attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
        position_ids = prompts.batch['position_ids']
        position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
        instance_ids = prompts.batch['instance_id']
        instance_ids = instance_ids.repeat_interleave(self.config.n, dim=0)
        processed_input_ids = []
        for prompt_token_ids in input_ids:
            processed_ids = _pre_process_inputs(self.pad_token_id, prompt_token_ids)
            processed_input_ids.append(processed_ids)
        raw_input_ids = input_ids
        input_ids = processed_input_ids
        input_device = prompts.batch['input_ids'].device
        
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        batch_size = prompts.batch['input_ids'].size(0) * self.config.n
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        max_seq_length = self.config.response_length + input_length
        max_turns = self.config.get('max_turns', 30)
        inference = MultiTurnInference(self, self.tokenizer, self.inference_engine, max_turns=max_turns)
        with self.update_sampling_params(**kwargs):
            # print("Call process_batch_prompts")
            sequences = inference.process_batch_prompts(self.sampling_params, input_ids, instance_ids, self.config.response_length)

        print(f"Generated {len(sequences)} responses out of {batch_size} prompts, original batch size is: {batch_size // self.config.n}")

        response_ids = [seq["response_ids"] for seq in sequences]
        response_ranges = [seq["response_ranges"] for seq in sequences]
        sids = [int(seq["sid"]) for seq in sequences]
        responses = [seq["response"] for seq in sequences]
        response_mask = torch.full((batch_size, self.config.response_length),
                                    0,
                                    dtype=torch.long, 
                                    device=input_device)
        final_input_ids = torch.full((batch_size, max_seq_length), 
                                    self.pad_token_id,
                                    dtype=torch.long, 
                                    device=input_device)
        final_attention_mask = torch.zeros((batch_size, max_seq_length),
                                    dtype=torch.long,
                                    device=input_device)

        for i, (response_id, attention_mask_value) in enumerate(zip(response_ids, attention_mask)):
            for response_range in response_ranges[i]:
                response_mask[i, response_range[0]: response_range[1]] = 1
            
            # add padding to the input_ids
            final_input_ids[i, :input_length] = raw_input_ids[i]
            final_input_ids[i, input_length:input_length+len(response_id)] = torch.tensor(response_id, dtype=torch.long, device=input_device)

            final_attention_mask[i, :input_length] = torch.tensor(attention_mask_value, dtype=torch.long, device=input_device)
            final_attention_mask[i, input_length:input_length+len(response_id)] = 1

        final_responses = torch.full((batch_size, self.config.response_length),
                                    self.pad_token_id,
                                    dtype=torch.long,
                                    device=input_device)
        for i, response in enumerate(responses):
            final_responses[i, :len(response)] = torch.tensor(response, dtype=torch.long, device=input_device)

        position_ids = torch.zeros_like(final_attention_mask, device=input_device)
        for i in range(batch_size):
            position_ids[i, :] = torch.cumsum(final_attention_mask[i, :], dim=0) - 1
            position_ids[i, final_attention_mask[i,:]==0] = 0  # it's fine because all the valid tokens a continuous

        batch = TensorDict({
                'prompts': raw_input_ids, # TODO: whether need to copy?
                'responses': final_responses,
                'input_ids': final_input_ids,
                'attention_mask': final_attention_mask,
                'position_ids': position_ids,
                'response_mask': response_mask,
                "sids": torch.tensor(sids, dtype=torch.int64, device=input_device),
                "instance_id": instance_ids,
            }, batch_size=len(sequences),
        )

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)