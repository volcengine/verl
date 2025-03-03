import asyncio
import os
from dataclasses import asdict

import httpx
import sglang as sgl
import torch
import torch.distributed
from sglang.srt.managers.io_struct import InitWeightsUpdateGroupReqInput, UpdateWeightsFromDistributedReqInput
from tensordict import TensorDict

from verl import DataProto
from verl.workers.rollout.base import BaseRollout


async def ids_agent_loop(prompt_ids, gen_fn, obs_fn):
    done = False
    all_ids = prompt_ids
    loss_mask = [0] * len(all_ids)
    obs_metrics = {}
    while not done:
        action = await gen_fn(all_ids)
        all_ids += action
        loss_mask += [1] * len(action)
        obs = await obs_fn(action)
        obs_ids = obs.pop("ids")
        all_ids += obs_ids
        loss_mask += [0] * len(obs["ids"])
        done = obs.pop("done")
        for k, v in obs.items():
            if k not in obs_metrics:
                obs_metrics[k] = v
            else:
                obs_metrics[k] += v
    return {
        "ids": all_ids,
        "loss_mask": loss_mask,
    }


class AsyncRollout(BaseRollout):
    def __init__(self, model_path, **kwargs):
        super().__init__()
        print(f"nodedup in AsyncRollout: {torch.distributed.is_initialized() = } {torch.distributed.get_rank() = }")
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        self.engine = sgl.Engine(model_path=model_path)
        print(f"nodedup {torch.distributed.get_rank() = } engine initialized")
        self.sampling_params = kwargs

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        tokenizer = self.engine.tokenizer_manager.tokenizer

        async def gen_fn(prompt):
            # if isinstance(prompt, str):
            #     text = await self.engine.async_generate(prompt=prompt, sampling_params=sampling_params)
            # else:
            assert isinstance(prompt, list) and isinstance(prompt[0], int), f"not list int: {prompt=}"
            text = await self.engine.async_generate(input_ids=prompt, sampling_params=sampling_params)
            return tokenizer.encode(text)

        # TODO: to support more generalized scenario, logics here should be decoupled into agent env.
        async def obs_fn(action_ids):
            # find <|observation|> token part
            stop_id = action_ids[-1]
            stop_token = tokenizer.decode([stop_id])
            print(f"stop token: [{stop_id}] - [{stop_token}]")
            # only finish with <|observation|> token can be multi-turn
            if not stop_token.strip() == '<|observation|>':
                return {"done": True, "ids": [], "observation_time": 0}
            action = tokenizer.decode(action_ids, skip_special_tokens=False)
            text = action.split("<|observation|>")[0].strip()

            # call api part
            url = "http://172.16.65.43:8888/observation_kilt/"
            # payload = {"content": text}
            # new feature, for kilt_browser, we currently use translator
            payload = {"content": text, "translate": True}
            try:
                # api_response = requests.post(url, json=payload)
                # return api_response.json()
                with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload)
                    ret = response.json()
            except Exception as e:
                print(f"API call failed: {e}")
                raise

            # combine part
            obv_combined = ['\n' + obv['content'].strip() for obv in ret]
            obs_text = f"{'<|observation|>'.join(obv_combined)}<|assistant|>\n"
            return {"done": False, "text": obs_text, "observation_time": 1}

        input_ids = prompts["input_ids"]
        device = input_ids.device
        tasks = [ids_agent_loop(prompt_ids=prompt, gen_fn=gen_fn, obs_fn=obs_fn) for prompt in input_ids]
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))

        # make batch
        max_len = max(len(r["ids"]) for r in results)
        all_ids = torch.zeros((len(results), max_len), dtype=torch.long, device=device)
        loss_mask = torch.zeros((len(results), max_len), dtype=torch.int, device=device)
        attn_mask = torch.zeros((len(results), max_len), dtype=torch.int, device=device)
        for i, r in enumerate(results):
            all_ids[i, :len(r["ids"])] = torch.tensor(r["ids"], device=device)
            loss_mask[i, :len(r["loss_mask"])] = torch.tensor(r["loss_mask"], device=device)
            attn_mask[i, :len(r["ids"])] = 1

        batch = TensorDict({
            "input_ids": all_ids,
            "loss_mask": loss_mask,
            "attention_mask": attn_mask,
            "observation_times": torch.zeros_like(all_ids),
        }, batch_size=len(results))

        return DataProto(batch=batch)

    def init_pg(self, update_group_args: InitWeightsUpdateGroupReqInput):
        self.engine.init_weights_update_group(**asdict(update_group_args))

    def update_weights(self, update_weight_args: UpdateWeightsFromDistributedReqInput):
        self.engine.update_weights_from_distributed(**asdict(update_weight_args))

