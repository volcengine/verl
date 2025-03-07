import asyncio
import os
from functools import partial

import sglang as sgl
import torch.distributed
from omegaconf import DictConfig
from sglang.srt.openai_api.adapter import v1_chat_generate_request, v1_chat_generate_response
from sglang.srt.openai_api.protocol import ChatCompletionRequest

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from .loops import *
from .tasks import *


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class AsyncRollout(BaseRollout):
    def __init__(self, model_path, config: DictConfig):
        super().__init__()
        torch.distributed.barrier()
        # print(f"nodedup in AsyncRollout: {torch.distributed.is_initialized() = } {torch.distributed.get_rank() = }")
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
        self.task_type = config.task_type
        self.sampling_params = dict(config.sampling_params)
        self.sampling_params.update({
            # "max_new_tokens": 512,
            "skip_special_tokens": False,
            "stop": ["<|user|>", "<|observation|>", "<|im_end|>"],
        })
        self.event_loop = asyncio.get_event_loop()

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        tokenizer = self.engine.tokenizer_manager.tokenizer

        async def gen_id(prompt):
            assert isinstance(prompt, list) and isinstance(prompt[0], int), f"not list int: {prompt=}"
            res = await self.engine.async_generate(input_ids=prompt, sampling_params=sampling_params)
            if torch.distributed.get_rank() == 0:
                print(f"nodedup {torch.distributed.get_rank()=} generated: {res=}")
            text = res["text"]
            finish_reason = res["meta_info"]["finish_reason"]
            if finish_reason["type"] == "stop":
                matched = finish_reason["matched"]
                if isinstance(matched, int):
                    matched = tokenizer.decode([matched])
                text += matched
            return tokenizer.encode(text)

        async def gen_chat(request):
            tokenizer_manager = self.engine.tokenizer_manager
            all_requests = [ChatCompletionRequest(**request)]
            adapted_request, request = v1_chat_generate_request(all_requests, tokenizer_manager)
            try:
                ret = await tokenizer_manager.generate_request(adapted_request).__anext__()
            except ValueError as e:
                print(f"Error generating chat: {e}")
                raise
            if not isinstance(ret, list):
                ret = [ret]

            message = v1_chat_generate_response(
                request,
                ret,
                cache_report=tokenizer_manager.server_args.enable_cache_report,
                tool_call_parser=tokenizer_manager.server_args.tool_call_parser,
                to_file=True,
            )[0]["body"]["choices"]["message"]

            return message

        n = self.config.n
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        input_ids = input_ids.repeat_interleave(n, dim=0)
        position_ids = prompts.batch["position_ids"].repeat_interleave(n, dim=0)
        attn_mask = prompts.batch["attention_mask"].repeat_interleave(n, dim=0)

        async def dr_start(index):
            return _pre_process_inputs(tokenizer.pad_token_id, input_ids[index])

        async def swedev_start(index):
            try:
                result = await asyncio.to_thread(initialize_runtime, index, prompts.batch['instance_id'][index // n].item())
                return {
                    "sid": result["sid"],
                    "sids": [result["sid"]], # will be treated as a obs metric, thus, will be gathered into batch, and later used in reward acquisition
                }
            except Exception as e:
                print(f"Error processing instance: {e}")
                # in original logic, mismatched sids count and instance_ids count will cause error eventually, better raise now
                raise

        # choose function set
        # TODO: maybe in init is better, but some functions are local
        # TODO: partial is not the best way to pass arguments
        loop_fn, start_fn, gen_fn, obs_fn, end_fn = {
            "dr": (ids_agent_loop, dr_start, gen_id, partial(dr_obs, tokenizer=tokenizer), dummy),
            "swedev": (ids_agent_loop, swedev_start, gen_id, partial(swe_dev_obs, tokenizer=tokenizer), swe_dev_end),
            "gen_chat": (openai_chat_agent_loop, partial(openai_chat_start, url="todo"), gen_chat, partial(openai_chat_obs, url="todo"), partial(openai_chat_end, url="todo")),
        }[self.task_type]

        # starting rollout
        device = input_ids.device
        print(f"In async rollout {self.config.max_turns=} {self.total_len=}")
        tasks = [loop_fn(
            index=i,
            start_fn=start_fn,
            gen_fn=gen_fn,
            obs_fn=obs_fn,
            end_fn=end_fn,
            max_turns=self.config.max_turns,
            max_length=self.total_len,
        ) for i in list(range(len(input_ids)))]
        results = self.event_loop.run_until_complete(asyncio.gather(*tasks))

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
            print(f'After prompt len: {prompt_len}, {len(r["loss_mask"])}')
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
            for k, v in r["obs_metrics"].items():
                if k not in obs_metrics:
                    obs_metrics[k] = []
                obs_metrics[k].append(v)
        print(f"{obs_metrics=}")
        obs_metrics = {k: torch.tensor(v, device=device) for k, v in obs_metrics.items()}

        # TODO: maybe put obs metrics into non_tensor_batch after resolving swedev "sids" placement problem
        batch = TensorDict({
            "prompts": input_ids,
            "responses": responses,
            "input_ids": concat_ids,
            "loss_mask": concat_loss_mask,
            "attention_mask": concat_attn_mask,
            "position_ids": position_ids,
            **obs_metrics,
        }, batch_size=batch_size)

        # torch.save(batch, f"batches/batch_{time.time()}.pt")

        return DataProto(batch=batch)
