import asyncio

import httpx
import traceback

from verl.utils.swedev_utils import *


async def dummy(*_, **__):
    pass


async def openai_chat_start(index, url):
    # TODO: exception handling in this function is tricky
    print(f"starting session {index=} @ {torch.distributed.get_rank()=}")
    async with httpx.AsyncClient() as client:
        response = await client.post(url + "/start_sample", json={
            "index": index,
            "session_id": index,
        }, timeout=30)
        ret = response.json()
    ret["sid"] = index
    return ret


async def openai_chat_obs(message, sid, url, **_):
    payload = {"messages": [message]}
    header = {"session_id": str(sid)}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url + "/interact", json=payload, headers=header, timeout=30)
            assert response.status_code == 200, f"Wrong status code: {response.status_code=} {response=}"
            ret = response.json()
            ret["failed_times"] = 0
    except Exception as e:
        print(f"API call failed: {e}")
        traceback.print_exc()
        ret = {"messages": [{"role": "user", "content": "Connection Error"}], "failed_times": 1, "finish": False}
    ret["observations_times"] = 1
    if torch.distributed.get_rank() == 0:
        print(f"Observation: {ret}")
    return ret


async def openai_chat_end(sid, done, url):
    if done:
        return
    payload = {"session_id": sid, "done": done}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url + "/cancel", json=payload)
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
