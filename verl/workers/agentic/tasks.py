import asyncio

import aiohttp
import traceback

import torch

from verl.utils.swedev_utils import *


async def dummy(*_, **__):
    pass


async def openai_chat_start(index, name, url):
    # TODO: exception handling in this function is tricky
    print(f"starting session {index=} @ {torch.distributed.get_rank()=}")
    if isinstance(index, torch.Tensor):
        index = index.item()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(url + "/start_sample", json={
            "index": index,
            "name": name,
        }) as response:
            ret = await response.json()
    ret["sid"] = response.headers["session_id"]
    return ret


async def openai_chat_obs(message, sid, url, **_):
    payload = {"messages": [message]}
    header = {"session_id": str(sid)}
    metrics = {"failed_times": 0, "observations_times": 1}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(url + "/interact", json=payload, headers=header) as response:
                assert response.status == 200, f"Wrong status code: {sid=} {response.status=} {await response.text()}"
                ret = await response.json()
                metrics["failed_times"] = 0
    except Exception as e:
        print(f"API call failed: {e}")
        traceback.print_exc()
        ret = {"messages": [{"role": "user", "content": "Connection Error"}], "finish": False, "reward": -1}
        metrics["failed_times"] += 1
    ret["metrics"] = ret.get("metrics", {}) | metrics
    if torch.distributed.get_rank() == 0:
        print(f"Observation: {ret}")
    return ret


async def openai_chat_end(sid, done, url):
    if done:
        return
    payload = {"session_id": sid, "done": done}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url + "/cancel", json=payload) as response:
                pass
    except Exception as e:
        print(f"API call failed when ending: {e}")


async def swe_dev_obs(action_ids, sid, tokenizer, **kwargs):
    action = tokenizer.decode(action_ids, skip_special_tokens=False)
    if is_stop(action):
        print(f"Action stop: {action}")
        return {"done": True, "ids": [], "observation_times": 0}

    result = call_observation_api(sid, action)
    # TODO(haoran): handle here
    try:
        obs = result["content"]
    except:
        obs = "Error"
    return {"done": False, "ids": tokenizer.encode(obs), "observation_times": 1}


async def swe_dev_end(sid, _done):
    await asyncio.to_thread(call_postprocess_api, sid)
