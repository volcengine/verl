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
    start_args: dict,
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
    start = await start_fn(**start_args)
    # TODO(haoran): pad here! （hanchen: padding prompt_ids may cause wrong generation...)
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
        "prompts": prompt_ids,
        "responses": all_ids[len(prompt_ids): max_length],
        "response_loss_mask": response_loss_mask,
        "obs_metrics": obs_metrics,
    }


async def openai_chat_agent_loop(
    start_args: dict,
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
    start = await start_fn(**start_args)
    print("start", start)
    history = start.pop("messages")
    tools = start.pop("tools")
    sid = start.pop("sid")
    collect_metrics(start.get("metrics", {}), obs_metrics)

    prompt_ids = tokenizer.apply_chat_template(history, tools=tools, tokenize=True)
    ids = []
    response_loss_mask = []

    last_len = len(prompt_ids)

    # interact
    # TODO: maybe keep track of tokens here, can provide early stopping feature
    for turn in range(max_turns):
        message = await gen_fn({"messages": history, "tools": tools})
        history.append(message)
        ids = tokenizer.apply_chat_template(history, tools=tools, tokenize=True)
        turn_ids = ids[last_len:]
        last_len = len(ids)
        response_loss_mask += [1] * len(turn_ids)

        obs = await obs_fn(message, sid)
        # possible injection here
        messages = obs.pop("messages")
        for message in messages:
            history.append(message)
            ids = tokenizer.apply_chat_template(history, tools=tools, tokenize=True)
            turn_ids = ids[last_len:]
            last_len = len(ids)
            if message["role"] == "assistant":
                response_loss_mask += [1] * len(turn_ids)
            else:
                response_loss_mask += [0] * len(turn_ids)

        done = obs.pop("finish")
        reward = obs.pop("reward")
        collect_metrics(obs.get("metrics", {}), obs_metrics)

        if done or len(ids) >= max_length:
            break

    await end_fn(sid, done)

    return {
        "prompts": prompt_ids,
        "responses": ids[len(prompt_ids): max_length + len(prompt_ids)],
        "response_loss_mask": response_loss_mask[:max_length],
        "reward": reward,
        "obs_metrics": obs_metrics,
    }
