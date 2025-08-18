# This file is adapted from multiple sources:
# THUDM/slime project
#    Original source: https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py
#    Original source: https://github.com/THUDM/slime/blob/main/slime/ray/rollout_data_source.py
#    Original source: https://github.com/THUDM/slime/blob/main/slime/ray/rollout.py
#    Copyright 2025 Zhipu AI
#    Licensed under the Apache License, Version 2.0

import asyncio
import threading
from collections import defaultdict
from typing import Dict, Union

import numpy as np
import ray
import torch
from sglang.srt.sampling.sampling_params import SamplingParams
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.buffer import Buffer, Status
from verl.workers.rollout.http_utils import get, post
from verl.workers.rollout.sglang_rollout.sglang_rollout import _post_process_outputs, _pre_process_inputs, _start_router


# from slime.utils.types import Dict
class AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        # Schedule a coroutine onto the loop and block until it's done
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


# Create one global instance
async_loop = None


def get_async_loop():
    global async_loop
    if async_loop is None:
        async_loop = AsyncLoopThread()
    return async_loop


def run(coro):
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)


@ray.remote
class RolloutManager:
    def __init__(self, config):
        self.config = config.actor_rollout_ref.rollout
        self.debug = True
        self.debug =False
        if self.debug:
            # self.sglang_router_ip, self.sglang_router_port = _start_router()
            self.sglang_router_ip = "127.0.0.1"
            self.sglang_router_port = 30000
        else:
            self.sglang_router_ip, self.sglang_router_port = _start_router()

        self.data_buffer = Buffer(config)
        self.partial_rollout = config.actor_rollout_ref.rollout.partial_rollout
        self.use_http2 = False
        self.n_samples_per_prompt = config.actor_rollout_ref.rollout.n
        self.over_sampling_batch_size = config.actor_rollout_ref.rollout.over_sampling_batch_size
        self.rollout_batch_size = config.actor_rollout_ref.rollout.rollout_batch_size
        local_path = config.actor_rollout_ref.model.path

        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True, use_fast=True)
        self.init_sampling_params()

        # Initialize missing attributes
        self.semaphore = asyncio.Semaphore(100)  # Limit concurrent requests
        self.group_rm = False  # Default group reward model setting
        self.use_token_output = True  # Default token output setting
        self.reset()

    def init_sampling_params(self):
        self.sampling_params = dict(
            max_new_tokens=self.config.response_length,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
        )

    def get_num_rollout_per_epoch(self):
        return self.data_buffer.get_num_rollout_per_epoch()

    def get_sglang_router_ip_and_port(self):
        return self.sglang_router_ip, self.sglang_router_port

    def reset(self):
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples_group: list[list[Dict]]):
        for group in samples_group:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    self.generate_group(
                        group,
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples_group)

    def rollout(self):
        rollout_result, aborted_samples = run(self.rollout_async())
        self.data_buffer.add_samples(aborted_samples)
        return self._convert_samples_to_data_proto(rollout_result)

    async def rollout_async(self) -> list[list[Dict]]:
        data = []
        pbar = tqdm(total=self.rollout_batch_size * self.n_samples_per_prompt, desc="Rollout generation")

        while len(data) < self.rollout_batch_size:
            while self.remaining_batch_size < self.rollout_batch_size:
                # 从buffer中获取样本并提交生成请求
                samples = self.data_buffer.get_samples(self.over_sampling_batch_size)

                self.submit_generate_tasks(samples)

            # 等待生成完成
            done, self.pendings = await asyncio.wait(self.pendings, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                group: list[Dict] = task.result()

                assert len(group) == self.n_samples_per_prompt
                # 过滤器相关逻辑去除

                # 添加样本到data
                if len(data) < self.rollout_batch_size:
                    data.append(group)
                    pbar.update(self.n_samples_per_prompt)

        pbar.close()
        print(f"Rollout generation finished, got {len(data)} samples", flush=True)

        aborted_samples = await self.abort()
        # aborted_samples = []
        print(f"Aborted {len(aborted_samples)} samples", flush=True)

        assert len(data) == self.rollout_batch_size, f"Got {len(data)} samples, expected {self.rollout_batch_size}"
        from loguru import logger as log

        log.info(f"Rollout generation finished, got {len(data)} samples")
        # data = sorted(data, key=lambda group: group[0].index)

        # 重置全局状态
        self.reset()
        return data, aborted_samples

    def _convert_samples_to_data_proto(self, samples: list[list[Dict]]):
        """
        Convert inference generated samples to training data.
        """
        output_req_list = sum(samples, [])

        idx = []
        position_ids = []

        # 批量收集所有请求中的张量
        idx = torch.cat([req["input_ids"].unsqueeze(0) for req in output_req_list], dim=0)
        attention_mask = torch.cat([req["attention_mask"].unsqueeze(0) for req in output_req_list], dim=0)
        position_ids = torch.cat([req["position_ids"].unsqueeze(0) for req in output_req_list], dim=0)
        response = pad_sequence(
            [torch.tensor(req["response"]) for req in output_req_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        if response.shape[-1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.tokenizer.pad_token_id)
        # import pdb; pdb.set_trace()

        batch_size = idx.size(0)
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=self.tokenizer.eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        # uids = np.array([req["uid"] for req in output_req_list], dtype=object)

        # Construct the batch data
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(output_req_list),
        )

        non_tensors = defaultdict(list)
        for req in output_req_list:
            for key, value in req.items():
                if isinstance(value, torch.Tensor):
                    pass
                else:
                    non_tensors[key].append(value)
        for key, val in non_tensors.items():
            non_tensors[key] = np.array(val, dtype=object)

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensors,
        )

    async def abort(
        self,
    ):
        aborted_samples = []

        assert not self.aborted
        self.aborted = True
        response = await get(
            f"http://{self.sglang_router_ip}:{self.sglang_router_port}/list_workers", use_http2=self.use_http2
        )

        # abort all the requests
        for url in response["urls"]:
            print(f"Abort request for {url}", flush=True)
            await post(f"{url}/abort_request", {"abort_all": True}, use_http2=False)

        # make sure all the pending tasks are finished
        count = 0
        while self.pendings:
            done, self.pendings = await asyncio.wait(self.pendings, return_when=asyncio.FIRST_COMPLETED)

            if not self.partial_rollout:
                continue

            # for partial rollout, collect the partial samples into the data buffer
            for task in done:
                group = task.result()

                # for sample in group:
                # if sample.response and "start_rollout_id" not in sample.metadata:
                #     sample.metadata["start_rollout_id"] = rollout_id
                aborted_samples += [group]
                count += len(group)

        if self.partial_rollout:
            print(f"Collected {count} partial samples into the data buffer", flush=True)
        # import pdb; pdb.set_trace()

        return aborted_samples

    async def generate(self, sample: dict) -> Dict:
        url = f"http://{self.sglang_router_ip}:{self.sglang_router_port}/generate"

        assert sample["status"] == Status.PENDING or sample["status"] == Status.ABORTED, (
            f"Dict status is {sample['status']}"
        )
        from loguru import logger as log

        new_sampling_params = self.sampling_params.copy()

        if len(sample["response"]) > 0:
            new_sampling_params["max_new_tokens"] -= len(sample["response"])

        assert new_sampling_params["max_new_tokens"] >= 0, (
            f"max_new_tokens: {new_sampling_params['max_new_tokens']} should not be less than 0"
        )
        if new_sampling_params["max_new_tokens"] == 0:
            sample["status"] = Status.TRUNCATED
            return sample

        # Prepare payload - shared structure
        payload = {
            "sampling_params": new_sampling_params,
            "return_logprob": self.use_token_output,
        }

        if self.use_token_output:
            if len(sample["response"]) > 0:
                input_token_ids = sample["raw_prompt_ids"] + sample["response"]
            else:
                input_token_ids = sample["raw_prompt_ids"]
            payload["input_ids"] = input_token_ids
        else:
            # String-based mode: original implementation
            input_text = sample["raw_prompt_ids"] + sample["response"]
            payload["text"] = input_text

        output = await post(url, payload, use_http2=self.use_http2)

        if "output_token_logprobs" in output["meta_info"]:
            new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            # Update sample with tokens directly
            sample["response_length"] += len(new_response_tokens)
            sample["response"] += new_response_tokens

        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample["status"] = Status.TRUNCATED
            case "abort":
                sample["status"] = Status.ABORTED
            case "stop":
                sample["status"] = Status.COMPLETED

        return sample

    async def generate_and_rm(self, sample: dict, evaluation=False) -> dict:
        # For samples with existing response, check if they're complete
        if sample["status"] == Status.COMPLETED or sample["status"] == Status.TRUNCATED:
            assert sample["response"] is not None
            # if not self.group_rm:
            #     assert sample.reward is not None
            return sample

        # generate
        async with self.semaphore:
            if self.aborted:
                sample["status"] = Status.ABORTED
                return sample

            sample = await self.generate(sample)

        if sample["status"] == Status.ABORTED:
            return sample

        return sample

    async def generate_group(self, group: list[Dict], evaluation=False) -> list[Dict]:
        if self.aborted:
            return group

        group = await asyncio.gather(*[self.generate_and_rm(sample, evaluation=evaluation) for sample in group])

        return group

    def generate_sequences(self, prompts: DataProto):
        return run(self.async_generate_sequences(prompts))

    async def async_generate_sequences(self, prompts: DataProto):
        """For compatibility with the original generate_sequences in verl"""
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]).tolist() for i in range(batch_size)],
                dtype=object,
            )

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
                strict=True,
            ):
                sglang_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                        "image_data": (
                            multi_modal_data.get("image", None) if isinstance(multi_modal_data, dict) else None
                        ),
                    }
                )
        else:
            sglang_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # Ensure token IDs are lists or numpy arrays
        for input_data in sglang_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )
        idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
        image_list = [input_data.get("image_data", None) for input_data in sglang_inputs]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": self.config.response_length,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )
        url = f"http://{self.sglang_router_ip}:{self.sglang_router_port}/generate"

        # 并发执行所有请求
        tasks = [
            post(
                url,
                {
                    "sampling_params": request_sampling_params,
                    "return_logprob": True,
                    "input_ids": idx,
                    "image_data": image_data,
                },
                use_http2=self.use_http2,
            )
            for idx, image_data in zip(idx_list, image_list)
        ]

        outputs = []
        pbar = tqdm(total=len(tasks), desc="Run eval Rollout generation")
        from loguru import logger as log
        for coro in asyncio.as_completed(tasks):
            output = await coro
            outputs.append(output)
            # log.info(f"output: {output}")
            pbar.update(1)
        pbar.close()

        out = _post_process_outputs(self.tokenizer, outputs)
        response = out[0].to(idx.device)
        rollout_log_probs = None
        if self.config.calculate_log_probs:
            rollout_log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.tokenizer.pad_token_id)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_sequence_to_length(
                    rollout_log_probs, self.config.response_length, self.tokenizer.pad_token_id
                )

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
