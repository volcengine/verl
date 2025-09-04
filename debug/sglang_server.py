import os

# bash /opt/tiger/mlx_deploy/mlx_launch_cmd.sh


os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""


def main():
    os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")

    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    kwargs = {
        "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "dtype": "bfloat16",
        "mem_fraction_static": 0.8,
        "enable_memory_saver": True,
        "base_gpu_id": 0,
        "gpu_id_step": 1,
        "tp_size": 2,
        "node_rank": 0,
        "dist_init_addr": None,
        "nnodes": 1,
        "trust_remote_code": False,
        "max_running_requests": 1024,
        "port": 30002,
        "log_level": "info",
        "mm_attention_backend": "fa3",
        "attention_backend": "fa3",
        "skip_tokenizer_init": True,
        "is_embedding": True,
    }
    server_args = ServerArgs(**kwargs)
    launch_server(server_args)

    exit()
    # server = AsyncHttpServerAdapter(**args)
    # tokenizer = AutoTokenizer.from_pretrained(args["model_path"])

    PROMPT = "What is the range of the numeric output of a sigmoid node in a neural network?"
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
