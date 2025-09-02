import os
from transformers import AutoTokenizer
from verl.workers.rollout.sglang_rollout.http_server_engine import HttpServerAdapter


# bash /opt/tiger/mlx_deploy/mlx_launch_cmd.sh


os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""


def main():
    args = {
        "model_path": "/mnt/hdfs/yyding/models/Skywork-Reward-Llama-3.1-8B-v0.2",
        "tp_size": 2,
        "first_rank_in_node": True,
        "is_embedding": True,
        "enable_memory_saver": True,
    }

    server = HttpServerAdapter(**args)
    tokenizer = AutoTokenizer.from_pretrained(args["model_path"])

    PROMPT = (
        "What is the range of the numeric output of a sigmoid node in a neural network?"
    )
    RESPONSE1 = "The output of a sigmoid node is bounded between -1 and 1."
    RESPONSE2 = "The output of a sigmoid node is bounded between 0 and 1."

    CONVS = [
        [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE1}],
        [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": RESPONSE2}],
    ]

    prompt = tokenizer.apply_chat_template(CONVS[0], tokenize=False)
    print(server.reward_score(prompt))

    # prompts = tokenizer.apply_chat_template(CONVS, tokenize=False)
    # print(prompts)
    server.release_memory_occupation()
    import pdb; pdb.set_trace()
    server.resume_memory_occupation()
    prompt = tokenizer.apply_chat_template(CONVS[0], tokenize=False)
    print(server.reward_score(prompt))

if __name__ == "__main__":
    main()
