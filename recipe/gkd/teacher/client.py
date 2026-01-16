# Copyright 2025 Individual Contributor: furunding
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
import queue
import random
import threading
from concurrent.futures import Future
from contextlib import nullcontext
from datetime import datetime

import torch
import zmq
from codetiming import Timer

try:
    from .utils import deserialize, serialize
except ImportError:
    from utils import deserialize, serialize

DEBUG = False


def check_if_invalid(topk_logps, inputs):
    is_valid = True
    reason = ""
    for x in topk_logps:
        if x.isnan().any():
            is_valid = False
            reason = "nan"
            break
        elif x.isinf().any():
            is_valid = False
            reason = "inf"
            break
        elif (x == 0).any():
            is_valid = False
            reason = "zero"
            break
    if not is_valid:
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.tolist()
        with open("teacher_debug.log", "a") as f:
            f.write("{}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            f.write(f"{reason}\n")
            f.write(f"{str(inputs)}\n")


class TeacherClient:
    def __init__(
        self,
        server_ip,
        server_port,
        num_microbatches=1,
        max_tokens=1,
        n_server_workers=1,
        temperature=1,
        only_response=False,
        max_seq_len=None,
    ) -> None:
        self.server_ip = server_ip
        self.server_port = server_port
        self.num_microbatches = num_microbatches
        self.n_server_workers = n_server_workers
        self.max_tokens = max_tokens
        self.task_queue = queue.Queue()
        self.mutex = threading.Lock() if n_server_workers > 1 else nullcontext()
        self.context = zmq.Context()
        self.temperature = temperature
        self.only_response = only_response
        self.max_seq_len = max_seq_len
        self._run()

    def bg_task(self):
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{self.server_ip}:{self.server_port}")
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, 600000)  # 接收超时 30 分钟

        while True:
            futures = []
            inputs = []
            batch = []
            try:
                with self.mutex:
                    for _ in range(self.num_microbatches):
                        future, data = self.task_queue.get()
                        if DEBUG:
                            inputs.append(data)
                        futures.append(future)
                        batch.extend(data.tolist() if isinstance(data, torch.Tensor) else data)

                if self.max_seq_len:
                    max_tokens = [min(self.max_tokens, self.max_seq_len - len(prompt)) for prompt in batch]
                    request = {"prompt_token_ids": batch, "max_tokens": max_tokens}
                else:
                    request = {"prompt_token_ids": batch, "max_tokens": self.max_tokens}
                if self.temperature:
                    request["temperature"] = self.temperature
                if self.only_response:
                    request["only_response"] = True

                socket.send(serialize(request))
                raw = socket.recv()
                response = deserialize(raw)

                if isinstance(response, dict) and response.get("status") == "error":
                    reason = response.get("reason", "unknown")
                    err = RuntimeError(f"Teacher error: {reason}")
                    for f in futures:
                        f.set_exception(err)
                    continue

                required = ("responses", "teacher_topk_logprobs", "teacher_topk_indices")
                for k in required:
                    if k not in response:
                        raise RuntimeError(f"Invalid response: missing key '{k}'")

                total = len(response["teacher_topk_logprobs"])
                if self.num_microbatches <= 0 or total % self.num_microbatches != 0:
                    raise RuntimeError(f"Size mismatch: total={total}, num_microbatches={self.num_microbatches}")

                mbs = total // self.num_microbatches
                for i, future in enumerate(futures):
                    s, e = i * mbs, (i + 1) * mbs
                    responses = response["responses"][s:e]
                    teacher_topk_logps = response["teacher_topk_logprobs"][s:e]
                    if DEBUG:
                        check_if_invalid(teacher_topk_logps, inputs[i])
                    teacher_topk_indices = response["teacher_topk_indices"][s:e]
                    future.set_result((responses, teacher_topk_logps, teacher_topk_indices))

            except zmq.Again:
                err = TimeoutError(f"Timeout waiting for server {self.server_ip}:{self.server_port}")
                for f in futures:
                    f.set_exception(err)
                continue
            except Exception as e:
                for f in futures:
                    try:
                        f.set_exception(e)
                    except Exception:
                        pass
                continue

    def _run(self):
        for _ in range(self.n_server_workers):
            threading.Thread(target=self.bg_task, daemon=True).start()

    def submit(self, data):
        future = Future()
        self.task_queue.put((future, data))
        return future

    def __del__(self):
        self.context.destroy()


if __name__ == "__main__":
    gbs = 128
    n_gps = 1
    mbs = 2
    seq_len = 4096

    prompt_lens = (n_gps * gbs) * [seq_len]

    tc = TeacherClient(
        server_ip="127.0.0.1", server_port=15555, num_microbatches=gbs // mbs, n_server_workers=1, only_response=False
    )

    prompt_token_ids = []

    for pl in prompt_lens:
        prompt_token_ids.append([random.randint(1, 99999) for j in range(pl)])

    with Timer(name="get_topk_logprobs", initial_text=True):
        futures = []
        for i in range(0, n_gps * gbs, mbs):
            futures.append(tc.submit(prompt_token_ids[i : i + mbs]))

        for future in futures:
            responses, teacher_topk_logprobs, teacher_topk_indices = future.result()

            print(len(teacher_topk_logprobs), len(teacher_topk_indices))

            assert len(responses) == mbs
            assert len(teacher_topk_logprobs) == mbs
            assert len(teacher_topk_indices) == mbs

            assert all(x.shape == y.shape for x, y in zip(teacher_topk_logprobs, teacher_topk_indices, strict=False))
            out_lens = [x.shape[0] for x in teacher_topk_logprobs]
            out_dims = [x.shape[1] for x in teacher_topk_logprobs]
            assert all(out_len == seq_len for out_len in out_lens)
            assert all(out_dim == 256 for out_dim in out_dims)
            assert all(x.dtype == torch.float32 for x in teacher_topk_logprobs), [
                x.dtype for x in teacher_topk_logprobs
            ]
            assert all(x.dtype == torch.int32 for x in teacher_topk_indices)
            assert all(x.dtype == torch.int32 for x in responses)
