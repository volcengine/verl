import os
from transformers import AutoTokenizer
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter


# bash /opt/tiger/mlx_deploy/mlx_launch_cmd.sh


os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""


def main():
    args = {
        "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "tp_size": 2,
        # "first_rank_in_node": True,
        "is_embedding": True,
        "enable_memory_saver": True,
    }

    from sglang.srt.server_args import ServerArgs
    from sglang.srt.entrypoints.http_server import launch_server
    server_args = ServerArgs(**args)
    launch_server(server_args)
    exit()
    # server = AsyncHttpServerAdapter(**args)
    # tokenizer = AutoTokenizer.from_pretrained(args["model_path"])

    PROMPT = (
        "What is the range of the numeric output of a sigmoid node in a neural network?"
    )
    RESPONSE1 = "The output of a sigmoid node is bounded between -1 and 1."
    RESPONSE2 = "The output of a sigmoid node is bounded between 0 and 1."

    CONVS = [
        [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE1}],
        [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE2}],
    ]

    # prompt = tokenizer.apply_chat_template(CONVS[0], tokenize=False)

    # prompts = tokenizer.apply_chat_template(CONVS, tokenize=False)
    # print(prompts)
    import asyncio
    prompt = tokenizer.apply_chat_template(CONVS[1], tokenize=False)
    loop = asyncio.get_event_loop()
    output = loop.run_until_complete(
        server.async_reward_score(
            prompt=prompt,
        )
    )
    print(output)

if __name__ == "__main__":
    main()
