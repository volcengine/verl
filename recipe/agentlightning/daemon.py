# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
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

import asyncio
import random
import socket
import threading
import time
import uuid
from typing import Optional

import numpy as np
import requests
import torch
from agentlightning import LLM, AgentLightningServer, NamedResources, Rollout, configure_logger
from flask import Flask, Response, abort, request
from tensordict import TensorDict

from verl import DataProto

configure_logger()


def get_left_padded_ids_and_attention_mask(ids: list[int], max_length: int, pad_token_id: int):
    """
    Left-pad (or truncate) a sequence of token IDs to a fixed length,
    and build the corresponding attention mask.

    Args:
        ids:             the original list of token IDs.
        max_length:      desired total length after padding/truncation.
        pad_token_id:    ID to use for padding.

    Returns:
        padded_ids:      list of length == max_length.
        attention_mask:  list of same length: 1 for non-pad tokens, 0 for pads.
    """
    seq_len = len(ids)

    if seq_len >= max_length:
        # too long → truncate from the left, keep the last max_length tokens
        trimmed = ids[-max_length:]
        attention_mask = [1] * max_length
        return trimmed, attention_mask

    # too short → pad on the left
    pad_len = max_length - seq_len
    padded_ids = [pad_token_id] * pad_len + ids
    attention_mask = [0] * pad_len + [1] * seq_len
    return padded_ids, attention_mask


def get_right_padded_ids_and_attention_mask(ids: list[int], max_length: int, pad_token_id: int):
    """
    Right-pad (or truncate) a sequence of token IDs to a fixed length,
    and build the corresponding attention mask.

    Args:
        ids:            the original list of token IDs.
        max_length:     desired total length after padding/truncation.
        pad_token_id:   ID to use for padding.

    Returns:
        padded_ids:     list of length == max_length.
        attention_mask: list of same length: 1 for non-pad tokens, 0 for pads.
    """
    seq_len = len(ids)

    if seq_len >= max_length:
        # too long → truncate to the first max_length tokens
        trimmed = ids[:max_length]
        attention_mask = [1] * max_length
        return trimmed, attention_mask

    # too short → pad on the right
    pad_len = max_length - seq_len
    padded_ids = ids + [pad_token_id] * pad_len
    attention_mask = [1] * seq_len + [0] * pad_len
    return padded_ids, attention_mask


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class AgentModeDaemon:
    """
    AgentModeDaemon using the AgentLightningServer SDK.

    This class manages the server lifecycle, task queueing, and results
    retrieval, while also running a proxy server for LLM requests. It maintains
    the original interface for compatibility with the RayPPOTrainer.
    """

    def __init__(
        self,
        port,
        train_rollout_n,
        train_information,
        tokenizer,
        mini_batch_size,
        pad_token_id,
        reward_fillna_value=0.0,
        llm_timeout_seconds=600.0,
    ):
        # Server and Task Configuration
        self.server_port = port
        self.llm_timeout_seconds = llm_timeout_seconds
        self.server = AgentLightningServer(
            host="0.0.0.0", port=self.server_port, task_timeout_seconds=self.llm_timeout_seconds
        )
        self.proxy_port = _find_available_port()  # Run proxy on a different port

        # Training and Data Configuration
        self.train_rollout_n = train_rollout_n
        self.train_information = train_information
        self.mini_batch_size = mini_batch_size
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.reward_fillna_value = reward_fillna_value

        # Internal State
        self.backend_llm_server_addresses: list[str] = []
        self._total_tasks_queued = 0
        self._completed_rollouts: dict[str, Rollout] = {}
        self._task_id_to_original_sample: dict[str, dict] = {}
        self._server_thread: Optional[threading.Thread] = None
        self._proxy_thread: Optional[threading.Thread] = None
        self.is_train = True

    def _start_proxy_server(self):
        """
        Initializes and runs a Flask-based proxy server in a separate thread.
        This proxy load-balances requests to the actual backend LLM servers.
        """
        app = Flask(__name__)

        num_requests = 0
        last_request_time = 0

        @app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
        def proxy(path):
            if not self.backend_llm_server_addresses:
                abort(503, description="No backend LLM servers available.")

            # Randomly choose a backend server for load balancing
            target_server = random.choice(self.backend_llm_server_addresses)
            target_url = f"http://{target_server}/v1/{path}"

            # Copy client request headers, removing the Host header
            headers = {key: value for key, value in request.headers if key.lower() != "host"}

            # Log the request for debugging
            nonlocal num_requests, last_request_time
            current_time = time.time()
            num_requests += 1
            if current_time - last_request_time > 60 or num_requests == 1 or num_requests % 100 == 0:
                print(f"Proxying {request.method} request to {target_server}. Request data: {request.get_data()}")
            last_request_time = current_time

            try:
                # Forward the request to the target backend
                resp = requests.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=request.args,
                    data=request.get_data(),
                    cookies=request.cookies,
                    allow_redirects=False,
                    timeout=self.llm_timeout_seconds,
                )
                # Filter out hop-by-hop headers before returning the response
                excluded_headers = [
                    "content-encoding",
                    "content-length",
                    "transfer-encoding",
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "upgrade",
                ]
                response_headers = [
                    (name, value) for name, value in resp.raw.headers.items() if name.lower() not in excluded_headers
                ]
                if resp.status_code == 200:
                    return Response(resp.content, status=resp.status_code, headers=response_headers)
                return Response(resp.content, resp.status_code, response_headers)
            except requests.exceptions.RequestException as e:
                abort(500, description=f"Error proxying request: {e}")

        def run_app():
            app.run(host="0.0.0.0", port=self.proxy_port, threaded=True, debug=False)

        self._proxy_thread = threading.Thread(target=run_app, daemon=True)
        self._proxy_thread.start()
        print(f"Proxy server running on port {self.proxy_port}")

    def start(self):
        """Starts the main AgentLightningServer and the proxy server."""

        def run_server():
            """Run the AgentLightningServer in a separate thread."""
            asyncio.run(self.server.run_forever())

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        # Wait for the server's internal startup event to be set.
        print("Waiting for AgentLightningServer to start...")
        is_ready = self.server.startup_event.wait(timeout=20.0)  # Wait up to 20s
        if not is_ready:
            raise RuntimeError("AgentLightningServer failed to start within the timeout period.")

        print(f"AgentLightningServer control plane running on port {self.server_port}")

        self._start_proxy_server()

    async def _async_set_up(self, data, server_addresses, is_train=True):
        """Async helper to set up data and resources on the server."""
        self.clear_data_and_server()
        self.backend_llm_server_addresses = server_addresses
        self.is_train = is_train

        # 1. Update resources on the server for clients to use
        llm_resource = LLM(
            endpoint=f"http://127.0.0.1:{self.proxy_port}/v1",
            model=self.train_information.get("model", "default-model"),
            sampling_parameters={"temperature": self.train_information.get("temperature", 0.7)},
        )
        resources: NamedResources = {"main_llm": llm_resource}
        resources_id = await self.server.update_resources(resources)

        # 2. Queue tasks for agents to process
        keys = list(data.keys())
        num_samples = len(data[keys[0]])
        rollouts_per_sample = self.train_rollout_n if is_train else 1

        for i in range(num_samples):
            data_id = str(uuid.uuid4())
            original_sample = {key: data[key][i] for key in keys}
            original_sample["data_id"] = data_id

            # For training, each sample is rolled out multiple times
            for j in range(rollouts_per_sample):
                task_metadata = {"data_id": data_id, "is_train": is_train}

                # Data ID is different from Rollout ID, as one data can have multiple rollouts.
                rollout_id = await self.server.queue_task(
                    sample=original_sample,
                    mode="train" if is_train else "val",
                    resources_id=resources_id,
                    metadata=task_metadata,
                )
                # Store original sample data to reconstruct batch information later
                self._task_id_to_original_sample[rollout_id] = original_sample
                self._total_tasks_queued += 1

    def set_up_data_and_server(self, data, server_addresses, is_train=True):
        """Synchronous wrapper for setting up data and server resources."""
        if not self.server.loop or not self.server.startup_event.is_set():
            raise RuntimeError("Server is not running or ready.")

        coro = self._async_set_up(data, server_addresses, is_train)
        future = asyncio.run_coroutine_threadsafe(coro, self.server.loop)
        try:
            future.result(timeout=60)  # Wait for completion with a timeout
        except Exception as e:
            print(f"Failed to set up data on server: {e}")
            raise

    async def _async_run_until_finished(self, verbose=True):
        """Async helper to wait for all tasks to complete."""
        while len(self._completed_rollouts) < self._total_tasks_queued:
            completed_batch = await self.server.retrieve_completed_rollouts()
            for rollout in completed_batch:
                self._completed_rollouts[rollout.rollout_id] = rollout
            if verbose:
                print(f"Completed {len(self._completed_rollouts)}/{self._total_tasks_queued} tasks...")
            await asyncio.sleep(5)
        print("All tasks finished.")

    def run_until_all_finished(self, verbose=True):
        """Synchronously waits for all queued tasks to be completed and reported."""
        if self._total_tasks_queued == 0:
            print("Warning: No tasks were queued.")
            return

        if not self.server.loop or not self.server.startup_event.is_set():
            raise RuntimeError("Server is not running or ready.")

        coro = self._async_run_until_finished(verbose)
        future = asyncio.run_coroutine_threadsafe(coro, self.server.loop)
        try:
            future.result()  # Wait indefinitely for all tasks to complete
        except Exception as e:
            print(f"Error while waiting for tasks to finish: {e}")
            raise

    def get_test_metrics(self):
        """Calculates and returns metrics for a validation run."""
        assert not self.is_train, "This method should only be called during validation."
        assert len(self._completed_rollouts) == self._total_tasks_queued

        sample_stat_list = []
        for rollout_id, rollout in self._completed_rollouts.items():
            if not rollout.triplets:
                continue
            response_length_list = [len(triplet.response.get("token_ids", [])) for triplet in rollout.triplets]
            final_reward = self._fillna_reward(rollout)
            sample_stat_list.append(
                {
                    "sum_response_length": np.sum(response_length_list),
                    "mean_response_length": np.mean(response_length_list) if response_length_list else 0,
                    "turn_count": len(rollout.triplets),
                    "reward": final_reward,
                }
            )

        return {
            "val/reward": np.mean([stat["reward"] for stat in sample_stat_list]),
            "val/mean_response_length": np.mean([stat["mean_response_length"] for stat in sample_stat_list]),
            "val/sum_response_length": np.mean([stat["sum_response_length"] for stat in sample_stat_list]),
            "val/turn_count": np.mean([stat["turn_count"] for stat in sample_stat_list]),
        }

    def get_train_data_batch(self, max_prompt_length, max_response_length, device):
        """
        Processes completed rollouts to generate a training data batch.

        This function reconstructs the logic from the original AgentModeDaemon,
        using data retrieved from the new server architecture. It handles padding,
        truncation, and tensor creation for the PPO training loop.
        """
        assert self.is_train, "This method should only be called during training."
        assert len(self._completed_rollouts) == self._total_tasks_queued

        # 1. Reconstruct the `finished_id_to_sample_info` structure from completed rollouts
        finished_id_to_sample_info = {}
        for rollout_id, rollout in self._completed_rollouts.items():
            original_sample = self._task_id_to_original_sample[rollout_id]

            if not rollout.triplets:
                continue

            # The client should report triplets that contain prompt_ids and response_ids.
            # Example triplet.prompt: {"token_ids": [...]}
            # Example triplet.response: {"token_ids": [...]}
            trace_list = [
                {"prompt_ids": t.prompt.get("token_ids", []), "response_ids": t.response.get("token_ids", [])}
                for t in rollout.triplets
            ]

            final_reward = self._fillna_reward(rollout)
            info = {
                "reward": final_reward,
                "trace_list": trace_list,
                "data_id": original_sample["data_id"],
            }
            finished_id_to_sample_info[rollout_id] = info
        #
        # --- Data processing and tensor creation logic ---
        # Get all the reported data.
        # prompt_ids are left-padded.
        # response_ids are right-padded.
        # They are concatenated in the middle.
        # Discard handling:
        #   - Those exceeding max_prompt_length will be marked for discard, but not
        #     discarded here. They are only truncated and marked, to be discarded later.
        #     This is for the correctness of the advantage calculation.
        #   - The discard for the PPO mini-batch should also be handled this way.
        input_ids_list, input_attention_mask_list = [], []
        response_ids_list, response_attention_mask_list = [], []
        reward_list, data_id_list, rollout_id_list, turn_index_list, is_drop_list = [], [], [], [], []
        n_trunc_sample_because_of_response = 0

        for rollout_id, sample_info in finished_id_to_sample_info.items():
            for turn_index, trace in enumerate(sample_info["trace_list"]):
                reward_list.append(sample_info["reward"])
                prompt_ids, response_ids = trace["prompt_ids"], trace["response_ids"]

                # Mark samples with prompts exceeding max_prompt_length to be dropped later
                if len(prompt_ids) > max_prompt_length:
                    prompt_ids = prompt_ids[:max_prompt_length]
                    is_drop_list.append(True)
                else:
                    is_drop_list.append(False)

                # Truncate responses that exceed max_response_length
                if len(response_ids) > max_response_length:
                    response_ids = response_ids[:max_response_length]
                    n_trunc_sample_because_of_response += 1

                # Pad prompts to the left and responses to the right
                one_input_ids, one_input_attention_mask = get_left_padded_ids_and_attention_mask(
                    prompt_ids, max_prompt_length, self.pad_token_id
                )
                one_response_ids, one_response_attention_mask = get_right_padded_ids_and_attention_mask(
                    response_ids, max_response_length, self.pad_token_id
                )

                input_ids_list.append(one_input_ids)
                input_attention_mask_list.append(one_input_attention_mask)
                response_ids_list.append(one_response_ids)
                response_attention_mask_list.append(one_response_attention_mask)
                data_id_list.append(sample_info["data_id"])
                rollout_id_list.append(rollout_id)
                turn_index_list.append(turn_index)

        n_transition = len(input_ids_list)
        batch_input_ids = torch.LongTensor(input_ids_list).to(device)
        input_attention_mask = torch.LongTensor(input_attention_mask_list).to(device)
        batch_response_ids = torch.LongTensor(response_ids_list).to(device)
        response_attention_mask = torch.LongTensor(response_attention_mask_list).to(device)

        # Concatenate prompts and responses to form the full sequence
        batch_seq = torch.cat([batch_input_ids, batch_response_ids], dim=-1)
        attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=-1)
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
        is_drop_mask = torch.BoolTensor(is_drop_list).to(device)
        scores = torch.tensor(reward_list, dtype=torch.bfloat16).to(device)

        # Create token-level scores by placing the final reward at the last token position
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)
        # At the eos_mask_idx position of each sample, fill in the corresponding scores.
        # torch.arange(n_transition) generates [0,1,2,...,bsz-1] as indices for the batch dimension.
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores[torch.arange(n_transition), eos_mask_idx] = scores
        # Only take the last response_length part of the sequence to get the token-level scores
        # for the model's response part.
        token_level_scores = token_level_scores[:, -max_response_length:]

        # Form the final batch using TensorDict
        batch = TensorDict(
            {
                "prompts": batch_input_ids,
                "responses": batch_response_ids,
                "input_ids": batch_seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "is_drop_mask": is_drop_mask,
                "token_level_scores": token_level_scores.contiguous(),
            },
            batch_size=n_transition,
        )
        data_proto = DataProto(batch=batch)

        data_metrics = {
            "agent_mode/n_trunc_sample_because_of_response": n_trunc_sample_because_of_response,
            "agent_mode/n_sample_to_train": n_transition,
        }

        # Add non-tensor data for advantage calculation and logging
        data_proto.non_tensor_batch["data_id_list"] = np.array(data_id_list)
        data_proto.non_tensor_batch["rollout_id_list"] = np.array(rollout_id_list)
        data_proto.non_tensor_batch["turn_index_list"] = np.array(turn_index_list)

        return data_proto, data_metrics

    def clear_data_and_server(self):
        """Resets the internal state of the daemon for the next run."""
        self.backend_llm_server_addresses = []
        self._completed_rollouts.clear()
        self._task_id_to_original_sample.clear()
        self._total_tasks_queued = 0
        # For a true reset, the server's internal queues would also need clearing.
        # This implementation assumes that `set_up_data_and_server` is called
        # for each new run, effectively starting a fresh batch.

    def _fillna_reward(self, rollout):
        if rollout.final_reward is None:
            if self.reward_fillna_value is not None:
                final_reward = self.reward_fillna_value
            else:
                raise ValueError(f"Reward is None for rollout {rollout.rollout_id}, please check the reward function.")
        else:
            final_reward = rollout.final_reward
        return final_reward
