import requests

# python -m sglang.launch_server --model-path /mnt/hdfs/yyding/models/Skywork-Reward-Llama-3.1-8B-v0.2 --is-embedding --port 30000
host = "127.0.0.1"
port = 30000
endpoint = "classify"
url = f"http://{host}:{port}/{endpoint}"
payload = {
    # "model": "/mnt/hdfs/yyding/models/Skywork-Reward-Llama-3.1-8B-v0.2",
    # "text": "What is the range of the numeric output of a sigmoid node in a neural network?\nThe output of a sigmoid node is bounded between -1 and 1.<eos>",
    "input_ids": [1, 2, 3]
}
response = requests.post(url, json=payload or {}, timeout=300)
print(response.json())

# 
