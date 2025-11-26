import requests

# vllm serve /home/tiger/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B --override-pooler-config '{"softmax": false}'
# llm = LLM(model="/home/tiger/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B")
# llm = LLM(model="/home/tiger/models/internlm/internlm2-1_8b-reward")
# output = llm.reward(["Hello, my name is"])


payloads = {
    "input": ["This product was excellent and exceeded my expectations"],
    "model": "/home/tiger/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B",
    "activation": False,
    "add_special_tokens": False,
}
response = requests.post("http://localhost:8000/classify", json=payloads)
outputs = response.json()
probs = [x["probs"] for x in outputs["data"]]
print(probs)
# [[0.9249725937843323]]


payloads = {
    "input": ["This product was excellent and exceeded my expectations"],
    "model": "/home/tiger/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B",
    "activation": False,
    "add_special_tokens": True,
}
response = requests.post("http://localhost:8000/classify", json=payloads)
outputs = response.json()
probs = [x["probs"] for x in outputs["data"]]
print(probs)
# [[0.9249725937843323]]


payloads = {
    "prompt": "This product was excellent and exceeded my expectations",
    "model": "/home/tiger/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B",
    "activation": False,
    "add_special_tokens": True,
}
response = requests.post("http://localhost:8000/tokenize", json=payloads)
outputs = response.json()
print(outputs)