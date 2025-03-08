# TODO(haoran): stuck in the loop
# TODO(haoran): time control; loss_mask
# TODO(haoran): check reason for loading weight
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


def _pre_process_inputs(pad_token_id, token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = token_ids[non_pad_index:].tolist()
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
            "skip_special_tokens": False,
        })
        self.event_loop = asyncio.get_event_loop()

    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
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

        # TODO: this is just a temporary approach for dr getting reward. should be moved to a backend.
        dr_storage_sid2seq = {}

        async def dr_start(index):
            dr_storage_sid2seq[index] = []
            return {"prompt_ids": _pre_process_inputs(tokenizer.pad_token_id, input_ids[index]), "sid": index}

        async def dr_obs(action_ids, sid, tokenizer, **_):
            # find <|observation|> token part
            dr_storage_sid2seq[sid].extend(action_ids)
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
            payload = {"content": text, "translate": True}
            failed = 0
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload)
                    ret = response.json()
            except Exception as e:
                print(f"API call failed: {e}")
                ret = [{"content": "API call failed"}]
                failed = 1

            # combine part
            obv_combined = ['\n' + obv['content'].strip() for obv in ret]
            obs_text = f"{'<|observation|>'.join(obv_combined)}<|assistant|>\n"
            ret_ids = tokenizer.encode(obs_text)
            dr_storage_sid2seq[sid].extend(ret_ids)
            return {"done": False, "ids": ret_ids, "observations_times": 1, "failed_times": failed}

        async def dr_end(sid, _):
            # currently for dr sid == index
            data_item = prompts[sid // n]
            from verl.utils.reward_score import _default_compute_score
            prompt_ids = data_item.batch['input_ids']
            prompt_length = prompt_ids.shape[-1]
            assert len(data_item.batch['attention_mask']) == prompt_length, f"{len(data_item.batch['attention_mask'])=} != {prompt_length=}"
            valid_prompt_length = data_item.batch['attention_mask'].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            sequences = dr_storage_sid2seq[sid]
            sequences_str = tokenizer.decode(sequences)
            prompt_str = tokenizer.decode(valid_prompt_ids)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            score = await asyncio.to_thread(_default_compute_score,
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                question=prompt_str,
                tokenizer=tokenizer,
            )
            return {
                "rm_final_scores": score,
            }

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
        loop_fn, start_fn, gen_fn, obs_fn, end_fn = {
            "dr": (ids_agent_loop, dr_start, gen_id, partial(dr_obs, tokenizer=tokenizer), dr_end),
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
        batch_size = len(results)
        pad = tokenizer.pad_token_id
        max_len, prompt_len, response_len = self.total_len, self.config.prompt_length, self.config.response_length
        prompts_ids = torch.full((batch_size, prompt_len), pad, dtype=torch.long, device=device)
        responses = torch.full((batch_size, response_len), pad, dtype=torch.long, device=device)
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.int, device=device)
        obs_metrics = {}

        for i, r in enumerate(results):
            prompts_ids[i, -len(r["prompts"]):] = torch.tensor(r["prompts"], device=device)
            length = min(len(r["responses"]), response_len)
            responses[i, :length] = torch.tensor(r["responses"], device=device)
            loss_mask[i, prompt_len: prompt_len + length] = torch.tensor(r["response_loss_mask"], device=device)

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
        }, batch_size=batch_size)

        # torch.save(batch, f"batches/batch_{time.time()}.pt")

        return DataProto(batch=batch)