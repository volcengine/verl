# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import requests
import soundfile as sf
import json
import numpy as np
import argparse
import time
import asyncio
import aiohttp

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8000",
        help="Address of the server",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="token2wav_asr",
        choices=[
            "token2wav_asr"
        ],
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--concurrent-job",
        type=int,
        default=10,
        help="Number of concurrent requests to send in parallel",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/emilia_zh-cosy-tiny-test.jsonl",
        help="Path to the data file",
    )
    return parser.parse_args()

def prepare_request(tokens, token_lens, gt_text):
    """Construct HTTP/JSON inference request body."""

    data = {
        "inputs": [
            {
                "name": "TOKENS",
                "shape": list(tokens.shape),
                "datatype": "INT32",
                "data": tokens.tolist(),
            },
            {
                "name": "TOKEN_LENS",
                "shape": list(token_lens.shape),
                "datatype": "INT32",
                "data": token_lens.tolist(),
            },
            {
                "name": "GT_TEXT",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [gt_text],
            },
        ]
    }

    return data

def load_jsonl(file_path: str):
    """Load data from jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


async def process_sample(idx, total, sample, session, url, semaphore):
    """Send a single request to the inference server and log the response."""
    async with semaphore:
        # Prepare request body
        code_list = sample["code"]
        tokens = np.array(code_list, dtype=np.int32).reshape(1, -1)
        token_lens = np.array([[len(tokens[0])]], dtype=np.int32)
        gt_text = sample["text"]
        data = prepare_request(tokens, token_lens, gt_text)

        # Send HTTP POST
        async with session.post(
            url,
            headers={"Content-Type": "application/json"},
            json=data,
            params={"request_id": "0"},
        ) as rsp:
            result = await rsp.json()

        # Parse outputs (order: REWARDS, TRANSCRIPTS)
        rewards = None
        transcripts = None
        for out in result.get("outputs", []):
            if out["name"] == "REWARDS":
                rewards = out["data"][0]
            elif out["name"] == "TRANSCRIPTS":
                transcripts = out["data"][0]

        # Output summary (prints may interleave across tasks)
        print(f"\n--- Sample {idx}/{total} ---")
        print(f"GT Text: {gt_text}")
        print(f"Tokens shape: {tokens.shape}, Token_lens shape: {token_lens.shape}")
        print(f"Transcript: {transcripts}")
        print(f"Reward: {rewards}")


async def main_async():
    args = get_args()

    server_url = args.server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"

    url = f"{server_url}/v2/models/{args.model_name}/infer"

    # Load dataset
    data_list = load_jsonl(args.data_path)

    # Concurrency primitives
    semaphore = asyncio.Semaphore(max(1, args.concurrent_job))
    connector = aiohttp.TCPConnector(ssl=False)

    start_time = time.time()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                process_sample(i + 1, len(data_list), sample, session, url, semaphore)
            )
            for i, sample in enumerate(data_list)
        ]
        await asyncio.gather(*tasks)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    asyncio.run(main_async())