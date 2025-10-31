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

import os

import zmq

context = zmq.Context()

frontend_listen_port = os.environ.get("PROXY_FRONTEND_PORT")
backend_listen_port = os.environ.get("PROXY_BACKEND_PORT")

assert frontend_listen_port is not None, "PROXY_FRONTEND_PORT is not set"
assert backend_listen_port is not None, "PROXY_BACKEND_PORT is not set"

# 创建前端 ROUTER 套接字并绑定到客户端连接地址
frontend = context.socket(zmq.ROUTER)
frontend.bind(f"tcp://*:{frontend_listen_port}")

# 创建后端 DEALER 套接字并绑定到服务端连接地址
backend = context.socket(zmq.DEALER)
backend.bind(f"tcp://*:{backend_listen_port}")

# 创建 poller 用于同时监听多个套接字
poller = zmq.Poller()
poller.register(frontend, zmq.POLLIN)
poller.register(backend, zmq.POLLIN)

print("proxy is running...")

while True:
    socks = dict(poller.poll())

    if frontend in socks:
        # 从 ROUTER 接收来自客户端的消息（multipart 消息）
        parts = frontend.recv_multipart()
        # print(f"收到客户端消息: {parts}")

        # 将完整的 multipart 消息转发给 DEALER
        backend.send_multipart(parts)

    if backend in socks:
        # 从 DEALER 接收来自服务端的回复
        reply_parts = backend.recv_multipart()
        # print(f"收到服务端回复: {reply_parts}")

        # 将回复转发回原始客户端（假设第一个部分是客户端 ID）
        frontend.send_multipart(reply_parts)
