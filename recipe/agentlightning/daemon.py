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
from agentlightning import LLM, AgentLightningServer, Rollout
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
        # Server configuration
        self.server_port = server_port
        self.proxy_port = proxy_port
        self.server = AgentLightningServer(host="0.0.0.0", port=server_port, task_timeout_seconds=300.0)

        # Training configuration
        self.train_rollout_n = train_rollout_n
        self.train_information = train_information
        self.mini_batch_size = mini_batch_size
        self.pad_token_id = pad_token_id

        # Runtime state
        self.backend_llm_server_addresses: list[str] = []
        self.is_train = True
        self._total_tasks_queued = 0
        self._completed_rollouts: dict[str, Rollout] = {}
        self._task_id_to_original_sample: dict[str, dict] = {}
        self._server_thread: Optional[threading.Thread] = None
        self._proxy_thread: Optional[threading.Thread] = None

    def _start_proxy_server(self):
        """Flask-based proxy server for load balancing LLM requests."""
        app = Flask(__name__)

        @app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
        def proxy(path):
            target_url = f"http://{random.choice(self.backend_llm_server_addresses)}/v1/{path}"
            headers = {k: v for k, v in request.headers if k.lower() != "host"}

            resp = requests.request(
                method=request.method,
                url=target_url,
                headers=headers,
                params=dict(request.args),
                data=request.get_data(),
                cookies=request.cookies,
                allow_redirects=False,
                timeout=300,
            )
            return Response(resp.content, resp.status_code, resp.headers.items())

        threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=self.proxy_port, threaded=True, debug=False), daemon=True
        ).start()
        print(f"Proxy server running on port {self.proxy_port}")

    def start(self):
        """Starts the AgentLightningServer and the proxy server."""
        self._server_thread = threading.Thread(target=lambda: asyncio.run(self.server.run_forever()), daemon=True)
        self._server_thread.start()
        self.server.startup_event.wait(timeout=20.0)
        print(f"AgentLightningServer running on port {self.server_port}")
        self._start_proxy_server()

    async def _async_set_up(self, data, server_addresses, is_train=True):
        """Async helper to set up data and resources on the server."""
        self.clear_data_and_server()
        self.backend_llm_server_addresses = server_addresses
        self.is_train = is_train

        # Update resources and queue tasks
        resources_id = await self.server.update_resources(
            {
                "main_llm": LLM(
                    endpoint=f"http://127.0.0.1:{self.proxy_port}/v1",
                    model=self.train_information.get("model", "default-model"),
                    sampling_parameters={"temperature": self.train_information.get("temperature", 0.7)},
                )
            }
        )

        keys = list(data.keys())
        rollouts_per_sample = self.train_rollout_n if is_train else 1

        for i in range(len(data[keys[0]])):
            original_sample = {key: data[key][i] for key in keys}
            original_sample["data_id"] = str(uuid.uuid4())

            for _ in range(rollouts_per_sample):
                rollout_id = await self.server.queue_task(
                    sample=original_sample, mode="train" if is_train else "val", resources_id=resources_id
                )
                self._task_id_to_original_sample[rollout_id] = original_sample
                self._total_tasks_queued += 1

    def set_up_data_and_server(self, data, server_addresses, is_train=True):
        """Synchronous wrapper for setting up data and server resources."""
        coro = self._async_set_up(data, server_addresses, is_train)
        future = asyncio.run_coroutine_threadsafe(coro, self.server.loop)  # type: ignore
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
        future = asyncio.run_coroutine_threadsafe(coro, self.server.loop)  # type: ignore
        future.result()

    def get_test_metrics(self):
        """Calculates and returns metrics for a validation run."""
        rollouts = list(self._completed_rollouts.values())
        return {
            "val/reward": np.mean([r.final_reward for r in rollouts if r.final_reward is not None]),
            "val/turn_count": np.mean([len(r.triplets) for r in rollouts if r.triplets]),
        }

    def get_train_data_batch(self, max_prompt_length, max_response_length, device):
        """
        Processes completed rollouts to generate a training data batch.
        """
        assert self.is_train, "This method should only be called during training."
        assert len(self._completed_rollouts) == self._total_tasks_queued

        batch_data = []

        for rollout_id, rollout in self._completed_rollouts.items():
            original_sample = self._task_id_to_original_sample[rollout_id]

            for turn_index, triplet in enumerate(rollout.triplets or []):
                prompt_ids = triplet.prompt["token_ids"][:max_prompt_length]
                response_ids = triplet.response["token_ids"][:max_response_length]

                input_ids, input_mask = get_left_padded_ids_and_attention_mask(
                    prompt_ids, max_prompt_length, self.pad_token_id
                )
                response_ids, response_mask = get_right_padded_ids_and_attention_mask(
                    response_ids, max_response_length, self.pad_token_id
                )

                batch_data.append(
                    {
                        "input_ids": input_ids,
                        "input_mask": input_mask,
                        "response_ids": response_ids,
                        "response_mask": response_mask,
                        "reward": rollout.final_reward,
                        "data_id": original_sample["data_id"],
                        "rollout_id": rollout_id,
                        "turn_index": turn_index,
                    }
                )

        # Convert to tensors
        input_ids_list = [d["input_ids"] for d in batch_data]
        input_attention_mask_list = [d["input_mask"] for d in batch_data]
        response_ids_list = [d["response_ids"] for d in batch_data]
        response_attention_mask_list = [d["response_mask"] for d in batch_data]
        reward_list = [d["reward"] for d in batch_data]
        data_id_list = [d["data_id"] for d in batch_data]
        rollout_id_list = [d["rollout_id"] for d in batch_data]
        turn_index_list = [d["turn_index"] for d in batch_data]

        n_transition = len(input_ids_list)
        batch_input_ids = torch.LongTensor(input_ids_list).to(device)
        input_attention_mask = torch.LongTensor(input_attention_mask_list).to(device)
        batch_response_ids = torch.LongTensor(response_ids_list).to(device)
        response_attention_mask = torch.LongTensor(response_attention_mask_list).to(device)

        # Concatenate prompts and responses to form the full sequence
        batch_seq = torch.cat([batch_input_ids, batch_response_ids], dim=-1)
        attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=-1)
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
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
