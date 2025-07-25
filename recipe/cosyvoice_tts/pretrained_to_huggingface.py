
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
Usage: Instruct TTS
  python3 infer.py \
    --token2wav-path /workspace/CosyVoice2-0.5B \
    --prompt-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
    --prompt-speech-path ./assets/prompt_audio.wav \
    --model-path ./transformers_cosyvoice2_llm \
    --input-text "用四川话说<|endofprompt|>扁担长，板凳宽，扁担绑在板凳上。吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。"
"""
from cosyvoice.cli.cosyvoice import CosyVoice2
import sys
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")



def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--pretrained-cosyvoice2-path",
        type=str,
        default="/workspace/CosyVoice2-0.5B",
        help="Token2Wav path, default to %(default)r",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default='./transformers_cosyvoice2_llm',
        help="The path to save the model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    cosy2_model = CosyVoice2(
        args.pretrained_cosyvoice2_path, load_jit=False, load_trt=False, fp16=False
    )

    llm = cosy2_model.model.llm.llm.model

    speech_embedding = cosy2_model.model.llm.speech_embedding
    llm_decoder = cosy2_model.model.llm.llm_decoder
    llm_embedding = cosy2_model.model.llm.llm_embedding

    tokenizer = AutoTokenizer.from_pretrained(f"{args.pretrained_cosyvoice2_path}/CosyVoice-BlankEN")
    special_tokens = {
        'eos_token': '<|endoftext|>',
        'pad_token': '<|endoftext|>',
        'additional_special_tokens': [
            '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
            '[breath]', '<strong>', '</strong>', '[noise]',
            '[laughter]', '[cough]', '[clucking]', '[accent]',
            '[quick_breath]',
            "<laughter>", "</laughter>",
            "[hissing]", "[sigh]", "[vocalized-noise]",
            "[lipsmack]", "[mn]"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

    original_tokenizer_vocab_size = len(tokenizer)
    cosyvoice2_token_size = 6561
    new_tokens = [f"<|s_{i}|>" for i in range(cosyvoice2_token_size)] + [
        "<|eos1|>", "<|eos2|>", "<|eos3|>", "<|sos|>", "<|task_id|>"
    ]
    num_added_tokens = tokenizer.add_tokens(new_tokens)

    llm.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)
    vocab_size = llm.get_input_embeddings().weight.shape[0]

    feature_size = speech_embedding.embedding_dim
    new_lm_head = torch.nn.Linear(in_features=feature_size, out_features=vocab_size, bias=True)

    with torch.no_grad():
        # set the weight and bias of the new lm_head to 0
        new_lm_head.weight.data.zero_()
        new_lm_head.bias.data.zero_()
        new_lm_head.weight[original_tokenizer_vocab_size:original_tokenizer_vocab_size+cosyvoice2_token_size+3] = llm_decoder.weight
        new_lm_head.bias[original_tokenizer_vocab_size:original_tokenizer_vocab_size+cosyvoice2_token_size+3] = llm_decoder.bias

    llm.lm_head = new_lm_head
    input_embeddings = llm.get_input_embeddings()

    with torch.no_grad():
        input_embeddings.weight[original_tokenizer_vocab_size:original_tokenizer_vocab_size+cosyvoice2_token_size+3] = speech_embedding.weight
        input_embeddings.weight[original_tokenizer_vocab_size+cosyvoice2_token_size+3:original_tokenizer_vocab_size+cosyvoice2_token_size+3+2] = llm_embedding.weight

    eos_token_ids = [original_tokenizer_vocab_size + cosyvoice2_token_size, original_tokenizer_vocab_size + cosyvoice2_token_size + 1, original_tokenizer_vocab_size + cosyvoice2_token_size + 2]
    llm.generation_config.eos_token_id = eos_token_ids
    llm.generation_config.temperature = 1.0
    llm.generation_config.top_p = 0.8
    llm.generation_config.top_k = 25

    llm.config.eos_token_id = original_tokenizer_vocab_size + cosyvoice2_token_size
    llm.config.vocab_size = vocab_size
    llm.config.tie_word_embeddings = False
    llm.config.use_bias = True
    llm.to(torch.bfloat16)
    llm.save_pretrained(args.save_path)

    TEMPLATE = "{%- for message in messages %}{%- if message['role'] == 'user' %}{{- '<|sos|>' + message['content'] + '<|task_id|>' }}{%- elif message['role'] == 'assistant' %}{{- message['content']}}{%- endif %}{%- endfor %}"
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(args.save_path)