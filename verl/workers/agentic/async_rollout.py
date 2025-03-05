import asyncio
import os
import time

import httpx
import sglang as sgl
import torch
import torch.distributed
from tensordict import TensorDict
from verl.utils.swedev_utils import *
from verl import DataProto
from verl.workers.rollout.base import BaseRollout
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

async def ids_agent_loop(prompt_ids, gen_fn, obs_fn, max_turns, max_length, sid=None):
    done = False
    all_ids = prompt_ids
    loss_mask = [0] * len(all_ids)
    obs_metrics = {}
    turn = 0
    while not done and len(all_ids) < max_length and turn < max_turns:
        action = await gen_fn(all_ids)
        all_ids += action
        loss_mask += [1] * len(action)
        if len(all_ids) >= max_length:
            break
        obs = await obs_fn(action, sid)
        obs_ids = obs.pop("ids")
        all_ids += obs_ids
        loss_mask += [0] * len(obs_ids)
        done = obs.pop("done")
        for k, v in obs.items():
            if k not in obs_metrics:
                obs_metrics[k] = v
            else:
                obs_metrics[k] += v
        turn += 1
    return {
        "ids": all_ids[:max_length],
        "loss_mask": loss_mask[:max_length],
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
        self.total_len = config.prompt_length + config.response_length
        print(f"async rollout {config.gpu_memory_utilization=}")
        self.engine = sgl.Engine(
            model_path=model_path,
            # cpu_offload_gb=500,
            port=40000,
            dtype=config.dtype,
            max_total_tokens=self.total_len,
            max_prefill_tokens=self.total_len,
            enable_memory_saver=True,
            mem_fraction_static=config.gpu_memory_utilization,
        )
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
        async def obs_fn(action_ids, sid=None):
            # find <|observation|> token part
            stop_id = action_ids[-1]
            stop_token = tokenizer.decode([stop_id])
            print(f"stop token: [{stop_id}] - [{stop_token}]")
            action = tokenizer.decode(action_ids, skip_special_tokens=False)
            if is_stop(action):
                return {"done": True, "ids": [], "observation_times": 0}
            obs = call_observation_api(sid, action)
            return {"done": False, "ids": tokenizer.encode(obs), "observation_times": 1}

        input_ids: torch.Tensor = prompts.batch["input_ids"]
        input_ids = input_ids.repeat_interleave(self.config.n, dim=0)
        instance_ids = None
        if self.config.is_swedev:
            instance_ids = prompts.batch['instance_id']
            instance_ids = instance_ids.repeat_interleave(self.config.n, dim=0)
        sids = [None] * len(input_ids)
        position_ids = prompts.batch["position_ids"].repeat_interleave(self.config.n, dim=0)
        attn_mask = prompts.batch["attention_mask"].repeat_interleave(self.config.n, dim=0)
        idx_list = [_pre_process_inputs(tokenizer.pad_token_id, input_ids[i]) for i in range(len(input_ids))]

        if self.config.is_swedev:
            with ThreadPoolExecutor(max_workers=min(len(instance_ids), 10)) as executor:
                future_to_idx = {executor.submit(initialize_runtime, idx, instance_id.item()): (idx, instance_id) for idx, instance_id in enumerate(instance_ids)}
                for future in as_completed(future_to_idx):
                    try:
                        result_idx, sid = future.result()
                        print(f"Got SID: {result_idx}, {sid}, SIDS: {sids}")
                        sids[result_idx] = sid
                    except Exception as e:
                        print(f"Error processing instance: {e}")
                        traceback.print_exc()

        device = input_ids.device
        print(f"In async rollout {self.config.max_turns=} {self.total_len=}")
        tasks = [ids_agent_loop(
            prompt_ids=prompt, 
            gen_fn=gen_fn,
            obs_fn=obs_fn, 
            max_turns=self.config.max_turns,
            max_length=self.total_len,
            sid=sid) for (prompt, sid) in zip(idx_list, sids)]
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*tasks))

        # make batch
        max_len = self.total_len
        # all_ids = torch.zeros((len(results), max_len), dtype=torch.long, device=device)
        responses = torch.zeros((len(results), max_len), dtype=torch.long, device=device)
        resp_loss_mask = torch.zeros((len(results), max_len), dtype=torch.int, device=device)
        resp_attn_mask = torch.zeros((len(results), max_len), dtype=torch.int, device=device)

        for i, r in enumerate(results):
            # all_ids[i, :len(r["ids"])] = torch.tensor(r["ids"], device=device)
            prompt_len = len(idx_list[i])
            gen_ids = torch.tensor(r["ids"][prompt_len:], device=device)
            resp_len = len(gen_ids)
            responses[i, :resp_len] = gen_ids
            resp_loss_mask[i, :len(r["loss_mask"]) - prompt_len] = torch.tensor(r["loss_mask"][prompt_len:], device=device)
            resp_attn_mask[i, :resp_len] = 1

        batch_size = len(results)
        response_length = responses.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        concat_ids = torch.cat([input_ids, responses], dim=1)
        concat_loss_mask = torch.cat([torch.zeros_like(input_ids), resp_loss_mask], dim=1)
        concat_attn_mask = torch.cat([attn_mask, resp_attn_mask], dim=1)

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
        sids = [sid if sid != None else 0 for sid in sids]
            
        batch = TensorDict({
            "prompts": input_ids,
            "responses": responses,
            "input_ids": concat_ids,
            "loss_mask": concat_loss_mask,
            "attention_mask": concat_attn_mask,
            "position_ids": position_ids,
            "sids": sids
            **obs_metrics,
        }, batch_size=batch_size)

        # torch.save(batch, f"batches/batch_{time.time()}.pt")

        return DataProto(batch=batch)
