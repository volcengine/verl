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

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from transformers import (
    ApertusConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    GemmaConfig,
    LlamaConfig,
    MistralConfig,
    Qwen2Config,
)

from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.utils.torch_functional import log_probs_from_logits_all_rmpad, masked_mean

# TODO(sgm): add more models for test
# we only need one scale for each model
test_configs = [
    LlamaConfig(num_hidden_layers=1),
    MistralConfig(num_hidden_layers=1),
    GemmaConfig(num_hidden_layers=1),
    Qwen2Config(num_hidden_layers=1),
    ApertusConfig(num_hidden_layers=1),
]


def test_hf_casual_models():
    batch_size = 4
    seqlen = 128
    response_length = 127

    for config in test_configs:
        # config = AutoConfig.from_pretrained(test_case)
        with torch.device("cuda"):
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
            model = model.to(device="cuda")
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device="cuda")
        attention_mask = create_random_mask(
            input_ids=input_ids,
            max_ratio_of_left_padding=0.1,
            max_ratio_of_valid_token=0.8,
            min_ratio_of_valid_token=0.5,
        )
        position_ids = compute_position_id_with_mask(
            attention_mask
        )  # TODO(sgm): we can construct the position_ids_rmpad here

        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # unpad the position_ids to align the rotary
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        logits_rmpad = model(
            input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False
        ).logits  # (1, total_nnz, vocab_size)

        origin_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
        ).logits
        origin_logits_rmpad, origin_logits_indices, *_ = unpad_input(origin_logits, attention_mask)

        logits_rmpad = logits_rmpad.squeeze(0)
        log_probs = log_probs_from_logits_all_rmpad(
            input_ids_rmpad=input_ids_rmpad,
            logits_rmpad=logits_rmpad,
            indices=indices,
            batch_size=batch_size,
            seqlen=seqlen,
            response_length=response_length,
        )  # (batch, seqlen)
        origin_log_probs = log_probs_from_logits_all_rmpad(
            input_ids_rmpad=input_ids_rmpad,
            logits_rmpad=origin_logits_rmpad,
            indices=origin_logits_indices,
            batch_size=batch_size,
            seqlen=seqlen,
            response_length=response_length,
        )  # (batch, seqlen)

        torch.testing.assert_close(
            masked_mean(log_probs, attention_mask[:, -response_length - 1 : -1]),
            masked_mean(origin_log_probs, attention_mask[:, -response_length - 1 : -1]),
            atol=1e-2,
            rtol=1e-5,
        )
    print("Check pass")


def test_hf_value_models():
    batch_size = 4
    seqlen = 128

    for config in test_configs:
        # config = AutoConfig.from_pretrained(test_case)
        config.num_labels = 1
        config.classifier_dropout = 0
        config.hidden_dropout = 0
        with torch.device("cuda"):
            model = AutoModelForTokenClassification.from_config(
                config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
            model = model.to(device="cuda")
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device="cuda")
        attention_mask = create_random_mask(
            input_ids=input_ids,
            max_ratio_of_left_padding=0.1,
            max_ratio_of_valid_token=0.8,
            min_ratio_of_valid_token=0.5,
        )
        position_ids = compute_position_id_with_mask(
            attention_mask
        )  # TODO(sgm): we can construct the position_ids_rmpad here

        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # unpad the position_ids to align the rotary
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        origin_logits = model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
        ).logits

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        rmpad_logits = model(
            input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False
        ).logits  # (1, total_nnz, 1)
        rmpad_logits = rmpad_logits.squeeze(0)
        pad_logits = pad_input(rmpad_logits, indices, batch_size, seqlen=seqlen)

        torch.testing.assert_close(
            masked_mean(pad_logits, attention_mask[:, :, None]),
            masked_mean(origin_logits, attention_mask[:, :, None]),
            atol=1e-2,
            rtol=1e-5,
        )
    print("Value model check pass")


if __name__ == "__main__":
    test_hf_casual_models()
    test_hf_value_models()
