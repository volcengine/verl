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
    CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path /mnt/hdfs/yyding/ckpts/MERGED_HF_MODEL/Qwen3-4B-Ins-GenRM-Step50
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

if __name__ == "__main__":
    check_health("http://127.0.0.1:55865/health")
