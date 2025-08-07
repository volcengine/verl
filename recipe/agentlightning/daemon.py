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
import threading
import uuid
from typing import Optional

import numpy as np
import requests
import torch
from agentlightning import LLM, AgentLightningServer, NamedResources, Rollout
from flask import Flask, Response, request
from tensordict import TensorDict
from utils import get_left_padded_ids_and_attention_mask, get_right_padded_ids_and_attention_mask

from verl import DataProto


class AgentManager:
    """
    Agent manager is built with Agent-Lightning SDK.

    This class manages the server lifecycle, task queueing, and results
    retrieval, while also running a proxy server for LLM requests. It maintains
    the original interface for compatibility with the RayPPOTrainer.
    """

    def __init__(
        self,
        server_port: int,
        proxy_port: int,
        train_rollout_n: int,
        train_information: dict,
        mini_batch_size: int,
        pad_token_id: int,
    ):
        self.server_port = server_port
        self.proxy_port = proxy_port
        self.llm_timeout_seconds = 300.0
        self.server = AgentLightningServer(
            host="0.0.0.0", port=self.server_port, task_timeout_seconds=self.llm_timeout_seconds
        )

        # Training and Data Configuration
        self.train_rollout_n = train_rollout_n
        self.train_information = train_information
        self.mini_batch_size = mini_batch_size
        self.pad_token_id = pad_token_id

        # Internal State
        self.backend_llm_server_addresses: list[str] = []
        self._total_tasks_queued = 0
        self._completed_rollouts: dict[str, Rollout] = {}
        self._task_id_to_original_sample: dict[str, dict] = {}
        self._server_thread: Optional[threading.Thread] = None
        self._proxy_thread: Optional[threading.Thread] = None
        self.is_train = True

    def _start_proxy_server(self):
        """Flask-based proxy server for load balancing LLM requests."""
        app = Flask(__name__)

        @app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
        def proxy(path):
            target_server = random.choice(self.backend_llm_server_addresses)
            target_url = f"http://{target_server}/v1/{path}"
            headers = {key: value for key, value in request.headers if key.lower() != "host"}

            resp = requests.request(
                method=request.method,
                url=target_url,
                headers=headers,
                params=request.args,
                data=request.get_data(),
                cookies=request.cookies,
                allow_redirects=False,
                timeout=300,
            )
            return Response(resp.content, resp.status_code, resp.headers.items())

        def run_app():
            app.run(host="0.0.0.0", port=self.proxy_port, threaded=True, debug=False)

        self._proxy_thread = threading.Thread(target=run_app, daemon=True)
        self._proxy_thread.start()
        print(f"Proxy server running on port {self.proxy_port}")

    def start(self):
        """Starts the main AgentLightningServer and the proxy server."""

        def run_server():
            asyncio.run(self.server.run_forever())

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()

        self.server.startup_event.wait(timeout=20.0)
        print(f"AgentLightningServer running on port {self.server_port}")
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
            for _ in range(rollouts_per_sample):
                rollout_id = await self.server.queue_task(
                    sample=original_sample,
                    mode="train" if is_train else "val",
                    resources_id=resources_id,
                )
                # Store original sample data to reconstruct batch information later
                self._task_id_to_original_sample[rollout_id] = original_sample
                self._total_tasks_queued += 1

    def set_up_data_and_server(self, data, server_addresses, is_train=True):
        """Synchronous wrapper for setting up data and server resources."""
        coro = self._async_set_up(data, server_addresses, is_train)
        future = asyncio.run_coroutine_threadsafe(coro, self.server.loop)
        future.result(timeout=60)

    async def _async_run_until_finished(self):
        """Async helper to wait for all tasks to complete."""
        while len(self._completed_rollouts) < self._total_tasks_queued:
            completed_batch = await self.server.retrieve_completed_rollouts()
            for rollout in completed_batch:
                self._completed_rollouts[rollout.rollout_id] = rollout
            print(f"Completed {len(self._completed_rollouts)}/{self._total_tasks_queued} tasks...")
            await asyncio.sleep(5)
        print("All tasks finished.")

    def run_until_all_finished(self):
        """Synchronously waits for all queued tasks to be completed and reported."""
        if self._total_tasks_queued == 0:
            return
        coro = self._async_run_until_finished()
        future = asyncio.run_coroutine_threadsafe(coro, self.server.loop)
        future.result()

    def get_test_metrics(self):
        """Calculates and returns metrics for a validation run."""
        return {
            "val/reward": np.mean([rollout.final_reward for rollout in self._completed_rollouts.values()]),
            "val/turn_count": np.mean([len(rollout.triplets) for rollout in self._completed_rollouts.values()]),
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

        input_ids_list, input_attention_mask_list = [], []
        response_ids_list, response_attention_mask_list = [], []
        reward_list, data_id_list, rollout_id_list, turn_index_list, is_drop_list = [], [], [], [], []

        for rollout_id, rollout in self._completed_rollouts.items():
            original_sample = self._task_id_to_original_sample[rollout_id]

            for turn_index, triplet in enumerate(rollout.triplets):
                prompt_ids = triplet.prompt["token_ids"]
                response_ids = triplet.response["token_ids"]

                reward_list.append(rollout.final_reward)

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
                data_id_list.append(original_sample["data_id"])
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

        # Add non-tensor data for advantage calculation and logging
        data_proto.non_tensor_batch["data_id_list"] = np.array(data_id_list)
        data_proto.non_tensor_batch["rollout_id_list"] = np.array(rollout_id_list)
        data_proto.non_tensor_batch["turn_index_list"] = np.array(turn_index_list)

        return data_proto

    def clear_data_and_server(self):
        """Resets the internal state of the daemon for the next run."""
        self.backend_llm_server_addresses = []
        self._completed_rollouts.clear()
        self._task_id_to_original_sample.clear()
        self._total_tasks_queued = 0
