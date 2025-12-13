# Copyright 2025 Individual Contributor: linxxx3 (linxxx3@gmail.com)
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
import time

import requests


def get_llm_server_address() -> str:
    """Get LLM server address from LLM_SERVER_NOTIFY_FILE."""
    notify_file = os.environ.get("LLM_SERVER_NOTIFY_FILE", None)
    if notify_file is None:
        raise ValueError("Please set LLM_SERVER_NOTIFY_FILE environment variable.")
    try:
        with open(notify_file) as f:
            lines = f.readlines()
            assert len(lines) > 0, f"{notify_file} is empty."
            server_address = lines[0].strip()
            return server_address
    except Exception as e:
        raise ValueError(f"Failed to read {notify_file}") from e


def notify_llm_server_address(address: str):
    notify_file = os.environ.get("LLM_SERVER_NOTIFY_FILE", None)
    if notify_file is None:
        raise ValueError("Please set LLM_SERVER_NOTIFY_FILE environment variable.")
    with open(notify_file, "w") as f:
        f.write(address)


def wait_for_server(address: str, url_path: str, timeout: int = 60):
    """Wait for the server to be ready."""
    if not address.startswith("http"):
        address = f"http://{address}"
    health_url = f"{address}{url_path}"
    print(f"Waiting for server at {health_url} ...")

    start_time = time.time()
    while True:
        try:
            response = requests.get(health_url, timeout=3)
            if response.status_code == 200:
                print(f"Server is ready at {address}")
                return
        except Exception:
            pass
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Server at {address} is not ready after {timeout} seconds.")
        time.sleep(1)
