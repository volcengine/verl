# TODO(haoran): stuck in the loop
# TODO(haoran): time control; loss_mask
# TODO(haoran): check reason for loading weight
import os
from functools import partial
from json import JSONDecodeError

import sglang as sgl
import torch.distributed
from omegaconf import DictConfig
from sglang.srt.function_call_parser import FunctionCallParser
from sglang.srt.openai_api.protocol import Tool
from torch.distributed import DeviceMesh

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from .loops import *
from .tasks import *


def _pre_process_inputs(pad_token_id, token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = token_ids[non_pad_index:].tolist()
    return token_ids

class AsyncRollout(BaseRollout):
    def __init__(self, model_path, config: DictConfig, device_mesh: DeviceMesh):
        super().__init__()
        torch.distributed.barrier()
        # print(f"nodedup in AsyncRollout: {torch.distributed.is_initialized() = } {torch.distributed.get_rank() = }")
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        self.tp_rank = device_mesh.get_local_rank(1)
        cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"]
        visible_devices: list[str | None] = [None] * device_mesh.size(1)
        torch.distributed.all_gather_object(visible_devices, cuda_visible_device, group=device_mesh.get_group(1))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        print(f"nodedup in async rollout {os.environ['CUDA_VISIBLE_DEVICES']=} @ {torch.distributed.get_rank()=} {self.tp_rank=}")
        self.total_len = config.prompt_length + config.response_length
        print(f"async rollout {config.gpu_memory_utilization=}")
        torch.distributed.barrier()
        # print(f"nodedup in async rollout {os.environ['CUDA_VISIBLE_DEVICES']=} @ {torch.distributed.get_rank()=} {self.tp_rank=}")
        if self.tp_rank == 0:
            self.engine = sgl.Engine(
                model_path=model_path,
                port=40000,
                dtype=config.dtype,
                max_total_tokens=self.total_len,
                max_prefill_tokens=self.total_len,
                enable_memory_saver=config.enable_memory_saver,
                mem_fraction_static=config.gpu_memory_utilization,
                tp_size=device_mesh.size(1),
                # enable_metrics=True,
            )
            print(f"nodedup {torch.distributed.get_rank() = } releasing memory occupation")
            self.engine.release_memory_occupation()
            print(f"nodedup {torch.distributed.get_rank() = } engine initialized")
        else:
            self.engine = None
        self.engine: sgl.srt.entrypoints.engine.Engine | None
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
        torch.distributed.barrier()
        self.config = config
        self.task_type = config.task_type
        self.sampling_params = dict(config.sampling_params)
        self.sampling_params.update({
            "skip_special_tokens": False,
        })
        self.event_loop = asyncio.get_event_loop()

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto | None:
        print(f"nodedup in generate seq {torch.distributed.get_rank()=} {self.tp_rank=} {prompts.batch['input_ids'].shape=} {prompts.non_tensor_batch=}")
        if self.tp_rank != 0:
            return None

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)
        tokenizer = self.engine.tokenizer_manager.tokenizer

        async def gen_id(input_ids):
            assert isinstance(input_ids, list) and isinstance(input_ids[0], int), f"not list int: {input_ids=}"
            res = await self.engine.async_generate(input_ids=input_ids, sampling_params=sampling_params)
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
            tools = request.get("tools", [])
            if torch.distributed.get_rank() == 0:
                print(f"generating request: {request}")
            ids = tokenizer.apply_chat_template(
                request["messages"],
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
            )
            try:
                ret = await self.engine.async_generate(input_ids=ids, sampling_params=sampling_params)
            except ValueError as e:
                print(f"Error generating chat: {e}")
                raise

            message = {
                "role": "assistant",
            }

            if tools:
                parser = FunctionCallParser(tools=[Tool.model_validate(tool) for tool in tools], tool_call_parser="qwen25")
                try:
                    normal_text, info_list = parser.parse_non_stream(ret["text"])
                except JSONDecodeError:
                    normal_text = ret["text"]
                    info_list = []
                message["content"] = normal_text
                message["tool_calls"] = [{
                    "id": str(info.tool_index),
                    "function": {
                        "name": info.name,
                        "arguments": info.parameters,
                    }
                } for info in info_list]
            else:
                message["content"] = ret["text"]

            if torch.distributed.get_rank() == 0:
                print(f"generated message: {message}")

            return message

        n = self.config.n
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        input_ids = input_ids.repeat_interleave(n, dim=0)

        # TODO: this is just a temporary approach for dr getting reward. should be moved to a backend.
        async def swedev_start(index):
            try:
                result = await initialize_runtime(prompts.batch['instance_id'][index // n].item())
                print(result)
                return {
                    "prompt_ids": _pre_process_inputs(tokenizer.pad_token_id, input_ids[index]),
                    "sid": result["sid"],
                    "sids": int(result["sid"]), # will be treated as a obs metric, thus, will be gathered into batch, and later used in reward acquisition
                }
            except Exception as e:
                # TODO: return true for handle api instead of raising an error
                print(f"Error processing instance: {e}")
                # in original logic, mismatched sids count and instance_ids count will cause error eventually, better raise now
                raise

        # choose function set
        # TODO: maybe in init is better, but some functions are local
        # TODO: partial is not the best way to pass arguments
        url = self.config["base_url"]
        loop_fn, start_fn, gen_fn, obs_fn, end_fn = {
            "swedev": (ids_agent_loop, swedev_start, gen_id, partial(swe_dev_obs, tokenizer=tokenizer), swe_dev_end),
            "gen_chat": (openai_chat_agent_loop, partial(openai_chat_start, url=url), gen_chat, partial(openai_chat_obs, url=url), partial(openai_chat_end, url=url)),
        }[self.task_type]

        # starting rollout
        device = input_ids.device
        print(f"In async rollout {self.config.max_turns=} {self.total_len=}")
        tasks = [loop_fn(
            index=torch.distributed.get_rank() * len(input_ids) + i,
            start_fn=start_fn,
            gen_fn=gen_fn,
            obs_fn=obs_fn,
            end_fn=end_fn,
            max_turns=self.config.max_turns,
            # max_length=self.total_len,
            # fix by lurui: consider some special token
            max_length=self.total_len - 50,
            tokenizer=tokenizer,
        ) for i in list(range(len(input_ids)))]
        results = self.event_loop.run_until_complete(asyncio.gather(*tasks))

        # make batch
        batch_size = len(results)
        pad = tokenizer.pad_token_id
        max_len, prompt_len, response_len = self.total_len, self.config.prompt_length, self.config.response_length
        prompts_ids = torch.full((batch_size, prompt_len), pad, dtype=torch.long, device=device)
        responses = torch.full((batch_size, response_len), pad, dtype=torch.long, device=device)
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.int, device=device)
        if "reward" in results[0]:
            rewards = torch.zeros((batch_size,), dtype=torch.float, device=device)
        obs_metrics = {}

        for i, r in enumerate(results):
            prompts_ids[i, -len(r["prompts"]):] = torch.tensor(r["prompts"], device=device)
            length = min(len(r["responses"]), response_len)
            responses[i, :length] = torch.tensor(r["responses"][:length], device=device)
            loss_mask[i, prompt_len: prompt_len + length] = torch.tensor(r["response_loss_mask"][:length], device=device)
            if "reward" in r:
                rewards[i] = r["reward"]

            for k, v in r["obs_metrics"].items():
                if k not in obs_metrics:
                    obs_metrics[k] = []
                obs_metrics[k].append(v)

        all_ids = torch.cat([prompts_ids, responses], dim=1)
        attn_mask = (all_ids != tokenizer.pad_token_id).int()
        position_ids = torch.zeros_like(attn_mask, device=device)
        for i in range(batch_size):
            position_ids[i, :] = torch.cumsum(attn_mask[i, :], dim=0) - 1
            position_ids[i, attn_mask[i, :] == 0] = 0  # it's fine because all the valid tokens a continuous

        print(f"{obs_metrics=}")
        obs_metrics = {k: torch.tensor(v, device=device) for k, v in obs_metrics.items()}

        # TODO: maybe put obs metrics into non_tensor_batch after resolving swedev "sids" & dr "rm_score" placement problem
        batch = TensorDict({
            "prompts": prompts_ids,
            "responses": responses,
            "input_ids": all_ids,
            "loss_mask": loss_mask,
            "attention_mask": attn_mask,
            "position_ids": position_ids,
            **obs_metrics,
            **({"rm_final_scores": rewards} if "reward" in results[0] else {}),
        }, batch_size=batch_size)

        return DataProto(batch=batch)