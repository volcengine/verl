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
    print("start", start)
    history = start.pop("messages")
    tools = start.pop("tools")
    sid = start.pop("sid")
    collect_metrics(start, obs_metrics)

    prompt_ids = tokenizer.apply_chat_template(history, tokenize=True)
    ids = []
    response_loss_mask = []

    # interact
    # TODO: maybe keep track of tokens here, can provide early stopping feature
    for turn in range(max_turns):
        message = await gen_fn({"messages": history, "tools": tools})
        turn_ids = tokenizer.apply_chat_template([message], tokenize=True)
        ids += turn_ids
        response_loss_mask += [1] * len(turn_ids)
        history.append(message)

        obs = await obs_fn(message, sid)
        # possible injection here
        messages = obs.pop("messages")
        for message in history:
            ids += turn_ids
            if message["role"] == "assistant":
                response_loss_mask += [1] * len(turn_ids)
            else:
                response_loss_mask += [0] * len(turn_ids)

        history += messages
        done = obs.pop("finish")
        reward = obs.pop("reward")
        collect_metrics(obs.get("metrics", {}), obs_metrics)

        if done or len(ids) >= max_length:
            break

    await end_fn(sid, done)

    return {
        "prompts": prompt_ids,
        "responses": ids[:max_length],
        "response_loss_mask": response_loss_mask[:max_length],
        "reward": reward,
        "obs_metrics": obs_metrics,
    }
