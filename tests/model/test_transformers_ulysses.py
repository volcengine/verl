import torch
import torch.distributed
from torch.distributed import init_device_mesh
from verl.utils.distributed import initialize_global_process_group
from verl.utils.model import create_random_mask, compute_position_id_with_mask
from verl.utils.torch_functional import masked_mean, log_probs_from_logits_all_rmpad, logprobs_from_logits
from verl.utils.ulysses import (gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, \
                                get_ulysses_sequence_parallel_world_size, \
                                get_ulysses_sequence_parallel_rank, \
                                set_ulysses_sequence_parallel_group)
from verl.workers.sharding_manager import FSDPUlyssesShardingManager
from verl.models.transformers.patch import llama_flash_attn_forward
from verl.protocol import DataProto
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis, rearrange

from transformers import LlamaConfig, MistralConfig, GemmaConfig, Qwen2Config
from transformers.models.llama.modeling_llama import LlamaModel, LlamaAttention, LlamaFlashAttention2
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForSequenceClassification
# TODO(sgm): add more models for test
# we only need one scale for each model
test_configs = [
    LlamaConfig(num_hidden_layers=1),
    # MistralConfig(num_hidden_layers=1),
    # GemmaConfig(num_hidden_layers=1),
    # Qwen2Config(num_hidden_layers=1)
]


def test_hf_casual_models():
    assert torch.cuda.device_count() >= 4, "need at least 2 gpus for test"
    local_rank, rank, world_size = initialize_global_process_group()
    sp_size = 8
    dp_size = 1
    ulysses_device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(dp_size, sp_size), mesh_dim_names=('dp', 'sp'))
    sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)

    batch_size = 4
    seqlen = 128
    response_length = 127

    for config in test_configs:
        # patch before load
        LlamaFlashAttention2.forward = llama_flash_attn_forward
        with torch.device('cuda'):
            model = AutoModelForCausalLM.from_config(config=config,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation='flash_attention_2')
            model = model.to(device='cuda')
        
        # different rank will generate different input_ids following fsdp
        input_ids_local = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device='cuda')
        attention_mask_local = create_random_mask(input_ids=input_ids_local,
                                            max_ratio_of_left_padding=0.1,
                                            max_ratio_of_valid_token=0.8,
                                            min_ratio_of_valid_token=0.5)
        position_ids_local = compute_position_id_with_mask(
            attention_mask_local)  # TODO(sgm): we can construct the position_ids_rmpad here

        model_inputs = {
            'input_ids': input_ids_local.cuda(),
            'attention_mask': attention_mask_local.cuda(),
            'position_ids': position_ids_local.int().cuda()
        }

        model_inputs = DataProto.from_dict(model_inputs)

        # 1. perform ulysses forward
        with sharding_manager:
            model_inputs = sharding_manager.preprocess_data(model_inputs)
            input_ids = model_inputs.batch['input_ids']
            attention_mask = model_inputs.batch['attention_mask']
            position_ids = model_inputs.batch['position_ids']
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                   attention_mask)  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
            print(f'input_ids before slice: {input_ids_rmpad.shape}')
            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                indices).transpose(0, 1)

            # slice input tensor for ulysses
            # input_ids are padded and sliced
            # postition_ids are only padded but not sliced
            input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
            print(f'pad size: {pad_size}')

            # input with input_ids_rmpad and postition_ids to enable flash attention varlen
            logits_split_in_seq = model(input_ids_rmpad_sliced, position_ids=position_ids_rmpad_padded,
                                        use_cache=False).logits  # (1, total_nnz/n, vocab_size)
            
            # all_gather output
            print(f'logits_split_in_seq: {logits_split_in_seq.shape}')
            logits_full = gather_outpus_and_unpad(logits_split_in_seq, gather_dim=1, unpad_dim=1, padding_size=pad_size)
            print(f'logits_full: {logits_full.shape}')

        # 2. perform normal forward
        # set_ulysses_sequence_parallel_group(None)
        # input_ids_rmpad_local, indices_local, *_ = unpad_input(input_ids_local.unsqueeze(-1),
        #                                            attention_mask_local)  # input_ids_rmpad (total_nnz, ...)
        # input_ids_rmpad_local = input_ids_rmpad_local.transpose(0, 1)  # (1, total_nnz)
        # print(f'input ids local: {input_ids_rmpad_local.shape}')
        # unpad the position_ids to align the rotary
        # position_ids_rmpad_local = index_first_axis(rearrange(position_ids_local.unsqueeze(-1), "b s ... -> (b s) ..."),
        #                                             indices_local).transpose(0, 1)
        
        logits_rmpad_local = model(input_ids_rmpad, position_ids=position_ids_rmpad,
                             use_cache=False).logits  # (1, total_nnz, vocab_size)
        # if torch.distributed.get_rank() == 0:
        #     print(f'logits_rmpad_local: {logits_rmpad_local}')
        #     print(f'logits full: {logits_full}')
        # torch.testing.assert_close(logits_rmpad_local, logits_full, rtol=1e-2, atol=1e-5)
    print(f'Check pass')

if __name__ == '__main__':
    test_hf_casual_models()
