import asyncio
import os
import time
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
        loss_mask += [0] * len(obs_ids)
        done = obs.pop("done")
        for k, v in obs.items():
            if k not in obs_metrics:
                obs_metrics[k] = v
            else:
                obs_metrics[k] += v
    return {
        "ids": all_ids,
        "loss_mask": loss_mask,
        **obs_metrics,
    }


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class AsyncRollout(BaseRollout):
    def __init__(self, model_path, config, **kwargs):
        super().__init__()
        torch.distributed.barrier()
        time.sleep(torch.distributed.get_rank() * 0)
        print(f"nodedup in AsyncRollout: {torch.distributed.is_initialized() = } {torch.distributed.get_rank() = }")
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        self.engine = sgl.Engine(model_path=model_path, cpu_offload_gb=500, enable_memory_saver=True, mem_fraction_static=0.3)
        print(f"nodedup {torch.distributed.get_rank() = } releasing memory occupation")
        self.engine.release_memory_occupation()
        print(f"nodedup {torch.distributed.get_rank() = } engine initialized")
        torch.distributed.barrier()
        self.config = config
        self.sampling_params = kwargs
        self.sampling_params.update({
            "max_new_tokens": 512,
            "skip_special_tokens": False,
            "stop": ["<|user|>", "<|observation|>", "<|im_end|>"],
        })

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        tokenizer = self.engine.tokenizer_manager.tokenizer

        async def gen_fn(prompt):
            # if isinstance(prompt, str):
            #     text = await self.engine.async_generate(prompt=prompt, sampling_params=sampling_params)
            # else:
            assert isinstance(prompt, list) and isinstance(prompt[0], int), f"not list int: {prompt=}"
            res = await self.engine.async_generate(input_ids=prompt, sampling_params=sampling_params)
            print(f"nodedup {torch.distributed.get_rank()=} generated: {res=}")
            text = res["text"]
            finish_reason = res["meta_info"]["finish_reason"]
            if finish_reason["type"] == "stop":
                matched = finish_reason["matched"]
                if isinstance(matched, int):
                    matched = tokenizer.decode([matched])
                text += matched
            return tokenizer.encode(text)

        # TODO: to support more generalized scenario, logics here should be decoupled into agent env.
        async def obs_fn(action_ids):
            # find <|observation|> token part
            stop_id = action_ids[-1]
            stop_token = tokenizer.decode([stop_id])
            print(f"stop token: [{stop_id}] - [{stop_token}]")
            # only finish with <|observation|> token can be multi-turn
            if not stop_token.strip() == '<|observation|>':
                return {"done": True, "ids": [], "observation_times": 0}
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
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload)
                    ret = response.json()
            except Exception as e:
                print(f"API call failed: {e}")
                ret = [{"content": "API call failed"}]
                # raise

            # combine part
            obv_combined = ['\n' + obv['content'].strip() for obv in ret]
            obs_text = f"{'<|observation|>'.join(obv_combined)}<|assistant|>\n"
            return {"done": False, "ids": tokenizer.encode(obs_text), "observation_times": 1}

        input_ids: torch.Tensor = prompts.batch["input_ids"]
        input_ids = input_ids.repeat_interleave(self.config.n, dim=0)
        position_ids = prompts.batch["position_ids"].repeat_interleave(self.config.n, dim=0)
        idx_list = [_pre_process_inputs(tokenizer.pad_token_id, input_ids[i]) for i in range(len(input_ids))]
        device = input_ids.device
        tasks = [ids_agent_loop(prompt_ids=prompt, gen_fn=gen_fn, obs_fn=obs_fn) for prompt in idx_list]
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))

        # make batch
        max_len = self.config.response_length
        # all_ids = torch.zeros((len(results), max_len), dtype=torch.long, device=device)
        responses = torch.zeros((len(results), max_len), dtype=torch.long, device=device)
        loss_mask = torch.zeros((len(results), max_len), dtype=torch.int, device=device)
        attn_mask = torch.zeros((len(results), max_len), dtype=torch.int, device=device)

        for i, r in enumerate(results):
            # all_ids[i, :len(r["ids"])] = torch.tensor(r["ids"], device=device)
            prompt_len = len(idx_list[i])
            gen_ids = torch.tensor(r["ids"][prompt_len:], device=device)
            resp_len = len(gen_ids)
            responses[i, :resp_len] = gen_ids
            loss_mask[i, :len(r["loss_mask"])] = torch.tensor(r["loss_mask"], device=device)
            attn_mask[i, :len(r["ids"])] = 1

        batch_size = len(results)
        response_length = responses.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        concat_ids = torch.cat([input_ids, responses], dim=1)

        # collect obs metrics
        obs_metrics = {}
        for r in results:
            for k, v in r.items():
                if k not in ["ids", "loss_mask"]:
                    if k not in obs_metrics:
                        obs_metrics[k] = []
                    obs_metrics[k].append(v)
        print(f"{obs_metrics=}")
        obs_metrics = {k: torch.tensor(v, device=device) for k, v in obs_metrics.items()}

        batch = TensorDict({
            "prompts": input_ids,
            "responses": responses,
            "input_ids": concat_ids,
            "loss_mask": loss_mask,
            "attention_mask": attn_mask,
            "position_ids": position_ids,
            **obs_metrics,
        }, batch_size=batch_size)

        return DataProto(batch=batch)
