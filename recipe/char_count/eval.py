from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = './models/sft/global_step_557'

with torch.device('cuda'):
    model = AutoModelForCausalLM.from_pretrained(path,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2')

tokenizer = AutoTokenizer.from_pretrained(path)

test_parquet = '/home/chi/data/char_count/sft/test.parquet'

import pandas as pd

test_data = pd.read_parquet(test_parquet)

prompt = test_data['prompt'][0]

prompt_with_chat_template = [{'role': 'user', 'content': prompt}]

prompt = tokenizer.apply_chat_template(prompt_with_chat_template, add_generation_prompt=True, tokenize=True,
                                       return_tensors='pt').to('cuda')

output = model.generate(prompt, max_new_tokens=512, do_sample=False)

output_str = tokenizer.batch_decode(output)

