from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path1 = "checkpoints/verl_func_rm_example_gsm8k/qwen2_5_0_5b_gen_rm/global_step_70/actor_merged_hf"
path2 = "checkpoints/verl_func_rm_example_gsm8k/qwen2_5_0_5b_gen_rm/global_step_20/actor_merged_hf"

prompt = "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

def completion(path, prompt): 
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
    
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=512)
    print(tok.decode(out[0], skip_special_tokens=True))

completion(path1, prompt)
completion(path2, prompt)
