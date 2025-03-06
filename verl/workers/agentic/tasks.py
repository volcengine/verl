import torch
import httpx
from verl.utils.swedev_utils import *


async def openai_chat_obs(message, sid, url, **_):
    payload = {"message": message}
    header = {"session_id": sid}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=header)
            ret = response.json()
            ret["failed_times"] = 0
    except Exception as e:
        print(f"API call failed: {e}")
        ret = {"messages": [{"role": "user", "content": "Connection Error"}], "failed_times": 1}
    ret["observations_times"] = 1
    return ret


async def dr_obs(action_ids, _sid, tokenizer, **_):
    # find <|observation|> token part
    stop_id = action_ids[-1]
    stop_token = tokenizer.decode([stop_id])
    print(f"stop token: [{stop_id}] - [{stop_token}]")

    # only finish with <|observation|> token can be multi-turn
    if not stop_token.strip() == '<|observation|>':
        return {"done": True, "ids": [], "observations_times": 0, "failed_times": 0}
    action = tokenizer.decode(action_ids, skip_special_tokens=False)
    text = action.split("<|observation|>")[0].strip()

    # call api part
    url = "http://172.16.65.43:8888/observation_kilt/"
    # payload = {"content": text}
    # new feature, for kilt_browser, we currently use translator
    payload = {"content": text, "translate": True}
    failed = 0
    try:
        # api_response = requests.post(url, json=payload)
        # return api_response.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            ret = response.json()
    except Exception as e:
        print(f"API call failed: {e}")
        ret = [{"content": "API call failed"}]
        failed = 1
        # raise

    # combine part
    obv_combined = ['\n' + obv['content'].strip() for obv in ret]
    obs_text = f"{'<|observation|>'.join(obv_combined)}<|assistant|>\n"
    return {"done": False, "ids": tokenizer.encode(obs_text), "observations_times": 1, "failed_times": failed}


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
