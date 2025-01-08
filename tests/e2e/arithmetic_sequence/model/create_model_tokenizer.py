# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Create a random model and tokenizer for PPO training
"""

import torch
import os
from transformers import AutoModelForCausalLM, LlamaConfig, AutoTokenizer

from tests.e2e.envs.digit_completion import CharTokenizer

tokenizer = CharTokenizer(
    characters=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', ':'],
    model_max_length=2048,
    chat_template=
    "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set role = message['role'] %}{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}{{ sep_token }}{% endif %}"
)

config = LlamaConfig(vocab_size=(tokenizer.vocab_size + 16 - 1) // 16 * 16,
                     hidden_size=128,
                     intermediate_size=344,
                     num_hidden_layers=4,
                     num_attention_heads=4,
                     num_key_value_heads=4,
                     pad_token_id=tokenizer.pad_token_id,
                     bos_token_id=tokenizer.bos_token_id,
                     eos_token_id=tokenizer.eos_token_id)

model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(model_folder, exist_ok=True)

model.save_pretrained(model_folder)

tokenizer_folder = model_folder
tokenizer.save_pretrained(tokenizer_folder)

load_tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder)

chat = [{'role': 'user', 'content': '1,0:2,3'}]

load_tokenizer.padding_side = 'left'
print(
    load_tokenizer.apply_chat_template(chat,
                                       tokenize=True,
                                       add_generation_prompt=True,
                                       max_length=10,
                                       padding='max_length'))
