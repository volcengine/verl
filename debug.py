import asyncio
import logging
import os
import socket
import multiprocessing

import uvicorn
from fastapi import FastAPI
from sglang import launch_server
from sglang_router.launch_router import RouterArgs, launch_router

logger = logging.getLogger(__file__)


def main():
    """
    CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path /mnt/hdfs/yyding/ckpts/MERGED_HF_MODEL/Qwen3-4B-Ins-GenRM-Step50 --port 30000
    CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path /mnt/hdfs/yyding/ckpts/MERGED_HF_MODEL/Qwen3-4B-Ins-GenRM-Step50 --port 30001
    python -m sglang_router.launch_router --worker-urls http://127.0.0.1:30000 http://127.0.0.1:30001 --port 11111
    """
    # ['127.0.0.1:51451', '127.0.0.1:46649']
    # router_args = RouterArgs(
    #     host="127.0.0.1",
    #     port=20000,
    #     eviction_interval=5,
    #     worker_urls=[
    #         'http://127.0.0.1:41119', 'http://127.0.0.1:62479',
    #     ],
    # )
    # router = launch_router(router_args)
    # import pdb; pdb.set_trace()
    # return router
    pass

import sys
import requests

def check_health(url="http://127.0.0.1:8000/health"):
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            print("✅ Server is healthy")
            return True
        else:
            print(f"❌ Server unhealthy: {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False

def generate(url="http://127.0.0.1:33959/generate"):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/mnt/hdfs/yyding/ckpts/MERGED_HF_MODEL/Qwen3-4B-Ins-GenRM-Step50")
    r = requests.post(url, json={"input_ids": [151644, 872, 198, 785, 2701, 374, 264, 6888, 3491, 448, 1181, 4910, 8046, 4226, 11, 3156, 448, 458, 15235, 6291, 320, 6960, 1119, 7354, 7731, 71111, 22079, 2533, 61686, 39911, 264, 738, 400, 32, 3, 315, 6785, 25780, 13, 5005, 14261, 11469, 678, 34226, 2477, 3194, 7289, 400, 33, 3, 315, 6785, 25780, 448, 279, 3343, 429, 279, 7192, 2392, 315, 400, 33, 3, 17180, 311, 400, 32, 12947, 14261, 594, 1140, 702, 220, 17, 15, 17, 19, 7289, 13, 7379, 279, 2629, 315, 279, 5424, 315, 362, 382, 58, 30714, 29098, 2533, 20, 20, 271, 58, 15469, 12478, 2533, 39807, 18762, 13, 362, 33968, 702, 264, 15138, 8482, 6896, 220, 16, 15, 15, 18762, 11, 678, 315, 2155, 4494, 13, 576, 2790, 897, 315, 279, 18762, 304, 279, 15138, 11, 979, 16099, 311, 39807, 18762, 11, 374, 220, 17, 15, 17, 19, 13, 2585, 1657, 6623, 18762, 1558, 279, 33968, 614, 304, 806, 15138, 1939, 5501, 2874, 3019, 553, 3019, 11, 323, 2182, 697, 1590, 4226, 2878, 1124, 79075, 6257, 382, 7771, 3383, 374, 311, 3395, 323, 42565, 279, 6291, 3019, 553, 3019, 13, 9646, 498, 10542, 458, 1465, 304, 264, 3019, 11, 470, 279, 1922, 315, 279, 3019, 1380, 279, 29658, 1465, 13666, 13, 18214, 11, 470, 279, 1922, 315, 481, 16, 320, 8206, 11136, 71114, 364, 1921, 1730, 863, 382, 5501, 2874, 3019, 553, 3019, 11, 2182, 697, 1590, 4226, 320, 72, 1734, 2572, 279, 1922, 8, 304, 1124, 79075, 46391, 151645, 198, 151644, 77091, 198]})
    print(tokenizer.decode(r.json()["output_ids"], skip_special_tokens=True))

if __name__ == "__main__":
    generate("http://127.0.0.1:43043/generate")
    # check_health("http://127.0.0.1:44631/health")
