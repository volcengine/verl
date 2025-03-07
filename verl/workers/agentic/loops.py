import re
from typing import Awaitable, Callable, Any

from transformers import PreTrainedTokenizerBase

SessionIdType = int
StarFnType = Callable[[int], Awaitable[dict]]
GenFnType = Callable[[Any], Awaitable]
ObsFnType = Callable[[Any, SessionIdType], Awaitable[dict]]
EndFnType = Callable[[int, bool], Awaitable]


def collect_metrics(src, tgt):
    for k, v in src.items():
        if k not in tgt:
            tgt[k] = v
        else:
            tgt[k] += v


async def ids_agent_loop(
    index: int,
    start_fn: StarFnType,
    gen_fn: GenFnType,
    obs_fn: ObsFnType,
    end_fn: EndFnType,
    max_turns: int,
    max_length: int,
    **_
) -> dict:
    done = False
    obs_metrics = {}
    start = await start_fn(index)
    # TODO(haoran): pad here!
    prompt_ids = start.pop("prompt_ids")
    all_ids = list(prompt_ids)
    sid = start.pop("sid")
    collect_metrics(start, obs_metrics)
    turn = 0
    response_loss_mask = []
    while not done and len(all_ids) < max_length and turn < max_turns:
        action = await gen_fn(all_ids)
        all_ids += action
        response_loss_mask += [1] * len(action)
        if len(all_ids) >= max_length:
            print(f"Too long... {len(all_ids)}, {max_length}")
            break
        obs = await obs_fn(action, sid)
        obs_ids = obs.pop("ids")
        all_ids += obs_ids
        response_loss_mask += [0] * len(obs_ids)
        done = obs.pop("done")
        collect_metrics(obs, obs_metrics)
        turn += 1
    collect_metrics(await end_fn(sid, done) or {}, obs_metrics)
    return {
        # TODO(haoran): return prompt_ids here
        "prompts": prompt_ids, 
        "responses": all_ids[len(prompt_ids):max_length],
        "response_loss_mask": response_loss_mask,
        "obs_metrics": obs_metrics,
    }


async def openai_chat_agent_loop(
    index: int,
    start_fn: StarFnType,
    gen_fn: GenFnType,
    obs_fn: ObsFnType,
    end_fn: EndFnType,
    max_turns: int,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    **_
) -> dict:
    done = False
    reward = 0
    obs_metrics = {}

    # start
    start = await start_fn(index)
    history = start.pop("history")
    sid = start.pop("sid")
    collect_metrics(start, obs_metrics)

    # interact
    # TODO: maybe keep track of tokens here, can provide early stopping feature
    for turn in range(max_turns):
        message = await gen_fn(history)
        history.append(message)
        obs = await obs_fn(message, sid)
        history += obs.pop("messages")
        done = obs.pop("done")
        collect_metrics(obs, obs_metrics)
        if done:
            break

    await end_fn(sid, done)

    # make ids and loss mask
    if re.search(r"\{\%-?\s*generation\s*-?\%\}", tokenizer.chat_template):
        # loss mask can be derived from chat template
        ids, loss_mask = tokenizer.apply_chat_template(
            history,
            tokenize=True,
            max_length=max_length,
            truncation=True,
            add_generation_prompt=True, # for glm models, generation prompt is eos
            return_assistant_tokens_mask=True,
        )
    else:
        ids = []
        loss_mask = []
        # TODO: add generation prompt for last turn
        for message in history:
            turn_ids = tokenizer.apply_chat_template([message], tokenize=True)
            ids += turn_ids
            if message["role"] == "assistant":
                loss_mask += [1] * len(turn_ids)
            else:
                loss_mask += [0] * len(turn_ids)

    return {
        "ids": ids[:max_length],
        "loss_mask": loss_mask[:max_length],
        "reward": reward,
        "obs_metrics": obs_metrics,
    }


# import traceback
# from concurrent.futures import ThreadPoolExecutor, as_completed

# async def ids_agent_loop(prompt_ids, gen_fn, obs_fn, max_turns, max_length, sid=None):
#     done = False
#     all_ids = prompt_ids
#     loss_mask = [0] * len(all_ids)
#     obs_metrics = {}
#     turn = 0
#     while not done and len(all_ids) < max_length and turn < max_turns:
#         action = await gen_fn(all_ids)
#         all_ids += action
#         loss_mask += [1] * len(action)
#         if len(all_ids) >= max_length:
#             print(f"Too long... {len(all_ids)}, {max_length}")
#             break
#         obs = await obs_fn(action, sid)
#         obs_ids = obs.pop("ids")
#         all_ids += obs_ids
#         loss_mask += [0] * len(obs_ids)
#         done = obs.pop("done")
#         for k, v in obs.items():
#             if k not in obs_metrics:
#                 obs_metrics[k] = v
#             else:
#                 obs_metrics[k] += v
#         turn += 1
#         await asyncio.sleep(3)

#     if sid: # for swedev postprocessing
#         await call_postprocess_api(sid)

#     return {
#         "ids": all_ids[:max_length],
#         "loss_mask": loss_mask[:max_length],
#         **obs_metrics,
#     }