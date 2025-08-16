import torch
from query_utils import create_advanced_query_understanding_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

query_text = "what is a famous travel destination in France?"

model_path = (
    "checkpoints/verl_func_rm_example_gsm8k/qwen2_5_0_5b_gen_rm_docleaderboard/"
    "global_step_20/actor/huggingface"
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)

# Use structured chat format - this is the key difference!
messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": create_advanced_query_understanding_prompt(query_text)},
]

# Apply chat template (this is what you were missing!)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Now tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)

# Decode only the new tokens (excluding the input prompt)
new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(response)
