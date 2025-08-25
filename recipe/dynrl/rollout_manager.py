import asyncio
import os
import socket
import threading
import time
import uuid
from collections import defaultdict
from typing import Dict

import numpy as np
import ray
import zmq
import zmq.asyncio
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils.net_utils import is_ipv6
from verl.utils.torch_functional import pad_2d_list_to_length


def get_host_ip():
    host_ipv4 = os.getenv("MY_HOST_IP", None)
    host_ipv6 = os.getenv("MY_HOST_IPV6", None)
    host_ip_by_env = host_ipv4 or host_ipv6
    return host_ip_by_env


def get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class ZMQBuffer:
    def __init__(self, zmq_port: int = 5555):
        self.zmq_port = zmq_port
        self.queue = asyncio.Queue()
        self.context = zmq.asyncio.Context()

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        asyncio.run_coroutine_threadsafe(self._zmq_server(), self.loop)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def get_server_addr(self):
        ip = get_host_ip()
        port = self.zmq_port
        return "tcp://" + f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"

    async def _zmq_server(self):
        socket = self.context.socket(zmq.PULL)
        if is_ipv6(get_host_ip()):
            socket.setsockopt(zmq.IPV6, 1)
        socket.bind(self.get_server_addr())
        while True:
            data = await socket.recv_json()
            await self.queue.put(data)

    async def get(self, timeout: float = None) -> Dict | None:
        try:
            if timeout is not None:
                return await asyncio.wait_for(self.queue.get(), timeout=timeout)
            return await self.queue.get()
        except asyncio.TimeoutError:
            return None


class RolloutManager:
    def __init__(self, config, worker_group: RayWorkerGroup, tokenizer) -> None:
        self.full_config = config
        self.config = config.actor_rollout_ref
        self.worker_group = worker_group
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        self.rollout_tp_size = self.config.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        self.zmq_buffer = ZMQBuffer(zmq_port=get_free_port())
        self.zmq_server_addr = self.zmq_buffer.get_server_addr()
        self.worker_group.setup_buffer(self.zmq_server_addr)
        print(f"ZMQ server addr: {self.zmq_server_addr}")

        self.prefix = "actor_rollout"

    def get_from_buffer(self, timeout: float = None):
        """Get a message from ZMQBuffer's queue in a synchronous context."""
        future = asyncio.run_coroutine_threadsafe(self.zmq_buffer.get(timeout=timeout), self.zmq_buffer.loop)
        try:
            return future.result(timeout + 1 if timeout else None)
        except Exception as e:
            print(f"Failed to get from buffer: {e}")
            return None

    def generate_sequences(self, prompts: DataProto, **sampling_params):
        assert self.zmq_buffer.queue.qsize() == 0, "ZMQ buffer is not empty."

        tik = time.perf_counter()
        self.worker_group.wake_up()

        batch_size = prompts.batch.batch_size[0]
        uids = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        prompts.non_tensor_batch["uid"] = uids
        if self.config.rollout.n > 1:
            batch_size = batch_size * self.config.rollout.n

        self.worker_group.generate_sequences(prompts)

        driver_info = {idx: self.rollout_tp_size for idx in range(0, self.worker_group.world_size, self.rollout_tp_size)}
        driver_worker_indices = sorted(driver_info.keys())
        # print(f"{driver_worker_indices=}")

        enable_dyn_scale = self.config.rollout.get("enable_dyn_scale", False)
        # get response from buffer
        results = []
        while len(results) < batch_size:
            # {uid: str, response: list[int]}
            item = self.get_from_buffer(timeout=1.0)
            if item is not None:
                results.append(item)

            # Check unfinished requests for all drivers
            num_unfinished_requests = ray.get([self.worker_group._workers[idx].actor_rollout_get_num_unfinished_requests.remote() for idx in driver_worker_indices])

            if enable_dyn_scale:
                # a naive dynamic scaling logic
                new_driver_worker_indices = []
                i = 0
                while i < len(driver_worker_indices):
                    current_idx = driver_worker_indices[i]

                    if i + 1 < len(driver_worker_indices):
                        next_idx = driver_worker_indices[i + 1]
                        if driver_info[current_idx] == driver_info[next_idx] and current_idx % (driver_info[current_idx] * 2) == 0 and num_unfinished_requests[i] + num_unfinished_requests[i + 1] <= 16 and num_unfinished_requests[i] + num_unfinished_requests[i + 1] > 0:
                            print(f"Scaling up drivers {current_idx} and {next_idx}, {num_unfinished_requests=} at {time.perf_counter() - tik:.2f} sec.", flush=True)

                            new_tp_size = driver_info[current_idx] * 2

                            worker_indices_to_merge = list(range(current_idx, current_idx + new_tp_size))

                            ray.get([self.worker_group._workers[idx].actor_rollout_scale_up.remote() for idx in worker_indices_to_merge])

                            driver_info[current_idx] = new_tp_size
                            new_driver_worker_indices.append(current_idx)

                            i += 2
                            continue

                    new_driver_worker_indices.append(current_idx)
                    i += 1

        self.worker_group.sleep()

        # reorder
        uid_to_responses = defaultdict(list)
        for item in results:
            uid_to_responses[item["uid"]].append(item["response"])
        ordered_responses = []
        for uid in uids:
            responses = uid_to_responses.get(uid, [])
            ordered_responses.extend(responses)
        ordered_responses = np.array(ordered_responses, dtype=object)

        assert len(ordered_responses) == batch_size, f"len(ordered_responses) != batch_size, {len(ordered_responses)=}, {batch_size=}(repeat:{self.config.rollout.n})"

        responses = pad_2d_list_to_length(ordered_responses, self.pad_token_id, max_length=self.config.rollout.response_length)
        gen_batch_output = TensorDict({"responses": responses}, batch_size=batch_size)
        gen_batch_output = DataProto(batch=gen_batch_output)

        prompts.non_tensor_batch.pop("uid", None)
        if self.config.rollout.n > 1:
            prompts = prompts.repeat(repeat_times=self.config.rollout.n, interleave=True)
        gen_batch_output = gen_batch_output.union(prompts)

        gen_batch_output = self.worker_group.postprocess_generate_sequences(gen_batch_output)

        # TODO: timing/reshard
        gen_batch_output.meta_info["timing"] = {}

        return gen_batch_output
