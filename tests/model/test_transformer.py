from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForTokenClassification, AutoTokenizer

import torch
from verl.utils.model import create_random_mask, compute_position_id_with_mask
from verl.utils.torch_functional import masked_mean
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis, rearrange


def test_hf_casual_models():
    batch_size = 4
    seqlen = 128

    # TODO(sgm): add more models for test
    # we only need one scale for each model
    test_cases = ['deepseek-ai/deepseek-llm-7b-chat', 'Qwen/Qwen2-7B-Instruct']
    for test_case in test_cases:
        config = AutoConfig.from_pretrained(test_case)
        model = AutoModelForCausalLM.from_pretrained(test_case,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation='flash_attention_2')
        model = model.to(device='cuda')
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device='cuda')
        attention_mask = create_random_mask(input_ids=input_ids,
                                            max_ratio_of_left_padding=0.1,
                                            max_ratio_of_valid_token=0.8,
                                            min_ratio_of_valid_token=0.5)
        position_ids = compute_position_id_with_mask(
            attention_mask)  # TODO(sgm): we can construct the position_ids_rmpad here

        input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
            input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # unpad the position_ids to align the rotary
        position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                              indices).transpose(0, 1)

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        rmpad_logits = model(input_ids_rmpad, position_ids=position_ids_rmpad,
                             use_cache=False).logits  # (1, total_nnz, vocab_size)
        pad_logits = pad_input(rmpad_logits.squeeze(0), indices, batch_size, seqlen=seqlen)

        origin_logits = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              position_ids=position_ids,
                              use_cache=False).logits

        torch.testing.assert_close(masked_mean(pad_logits, attention_mask[:, :, None]),
                                   masked_mean(origin_logits, attention_mask[:, :, None]),
                                   msg=f'{test_case} rmpad and non-rmpad logits are not equal')
    print(f'Check pass')


def test_hf_value_models():
    batch_size = 4
    seqlen = 128

    # TODO(sgm): add more models for test
    # we only need one scale for each model
    test_cases = ['deepseek-ai/deepseek-llm-7b-chat', 'Qwen/Qwen2-7B-Instruct']
    for test_case in test_cases:
        config = AutoConfig.from_pretrained(test_case)
        config.num_labels = 1
        setattr(config, 'classifier_dropout', 0)
        setattr(config, 'hidden_dropout', 0)
        model = AutoModelForTokenClassification.from_pretrained(test_case,
                                                                torch_dtype=torch.bfloat16,
                                                                config=config,
                                                                attn_implementation='flash_attention_2')
        model = model.to(device='cuda')
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device='cuda')
        attention_mask = create_random_mask(input_ids=input_ids,
                                            max_ratio_of_left_padding=0.1,
                                            max_ratio_of_valid_token=0.8,
                                            min_ratio_of_valid_token=0.5)
        position_ids = compute_position_id_with_mask(
            attention_mask)  # TODO(sgm): we can construct the position_ids_rmpad here

        input_ids_rmpad, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
            input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # unpad the position_ids to align the rotary
        position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                              indices).transpose(0, 1)

        origin_logits = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              position_ids=position_ids,
                              use_cache=False).logits
        print(f'origin_logits: {origin_logits.shape}')

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        rmpad_logits = model(input_ids_rmpad, position_ids=position_ids_rmpad,
                             use_cache=False).logits  # (1, total_nnz, vocab_size)
        print(f'rmpad_logits: {rmpad_logits.shape}')
        pad_logits = pad_input(rmpad_logits.squeeze(0), indices, batch_size, seqlen=seqlen)

        torch.testing.assert_close(masked_mean(pad_logits, attention_mask[:, :, None]),
                                   masked_mean(origin_logits, attention_mask[:, :, None]),
                                   msg=f'{test_case} rmpad and non-rmpad logits are not equal')
    print('Value model check pass')


if __name__ == '__main__':
    # test_hf_casual_models()
    test_hf_value_models()
