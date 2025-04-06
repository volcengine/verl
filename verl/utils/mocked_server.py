import logging
import socket
import multiprocessing
import queue
import uuid
import time
import threading
from typing import List
from collections import defaultdict

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    messages: List[dict]
    model: str


class TaskScore(BaseModel):
    request_id: str
    score: float


def get_ip():
    try:
        # 获取本地主机名
        hostname = socket.gethostname()
        # 获取本地 IP 地址
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except socket.gaierror as e:
        print(f"获取本地 IP 地址时出错: {e}")
        return None


class MockedServer:

    def __init__(self, port, ip, notify_score_fn=None):
        self.app = FastAPI()
        self.port = port
        self.notify_score_fn = notify_score_fn

        self.prompt_queue = multiprocessing.Queue()
        self.response_dict = {}
        self.rounds_dict = defaultdict(int)

        self.ip = ip
        assert self.ip is not None, f"ip for llm_base_url cannot be None"
        self.llm_base_url = (f"[{self.ip}]" if ":" in self.ip else f"{self.ip}") + f":{self.port}"

        self._define_routes()

    def _define_routes(self):
        # mocked openai api for chat completion
        @self.app.post("/v1chat/completions")
        def chat(request: ChatCompletionRequest):
            request_id = str(uuid.uuid4()) if request.model == "" else request.model
            self.rounds_dict[request_id] += 1
            self.add_response_endpoint(request_id)
            return self.chat(request_id, request)

        @self.app.post("/scores")
        def score(request: TaskScore):
            if self.notify_score_fn is not None:
                request_id = request.request_id
                round_count = self.rounds_dict.pop(request_id)
                self.notify_score_fn(request_id, request.score, round_count)

    def chat(self, request_id: str, request: ChatCompletionRequest):
        self.prompt_queue.put((request_id, request))
        return {"id": request_id, "object": "chat.completion", "model": "", "created": int(time.time()), "choices": []}

    def get_result(self, request_id: str):
        if request_id in self.response_dict:
            raw_result = self.response_dict.pop(request_id)
            logprobs = raw_result["logprobs"]
            response = raw_result["response"]
            input_ids = raw_result["input_ids"]

            result = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response,
                        "reasoning_content": "",
                    },
                    "finish_reason": "stop",
                    "logprobs": logprobs,
                    "input_ids": input_ids,
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            }
            return result
        else:
            return {}

    def add_response_endpoint(self, request_id):
        full_endpoint = f"/v1chat/completions/{request_id}"
        self.app.add_api_route(full_endpoint, self.get_result, methods=["GET"], name=request_id)

    def update_result(self, batched_results):
        self.response_dict.update(batched_results)


class ServerWithTask(MockedServer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_queue = queue.Queue()

    def _add_task_route(self):

        @self.app.get("/tasks")
        def get_task():
            try:
                result = self.task_queue.get(block=False)
                return result
            except Exception as e:
                return {"task": {}, "error": "No tasks available"}


def create_and_launch_task_pool(port, ip=None, notify_score_fn=None, with_task=False):
    server_cls = ServerWithTask if with_task else MockedServer
    mocked_server = server_cls(port=port, ip=ip, notify_score_fn=notify_score_fn)

    if ip is None:
        ip = get_ip()

    # launch Uvicorn server thread
    logging.info(f"using port: {mocked_server.port}")
    server_thread = threading.Thread(target=lambda: uvicorn.run(mocked_server.app,
                                                                host="::" if ":" in ip else "0.0.0.0",
                                                                port=mocked_server.port,
                                                                limit_concurrency=2000,
                                                                log_level="info"))
    server_thread.daemon = True
    server_thread.start()

    return mocked_server, server_thread
