import torch
import torch.distributed
from torch.distributed import init_device_mesh
from verl.utils.distributed import initialize_global_process_group
from verl.utils.model import create_random_mask, compute_position_id_with_mask
from verl.utils.torch_functional import masked_mean, log_probs_from_logits_all_rmpad, logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, get_ulysses_sequence_parallel_world_size
from verl.workers.sharding_manager import FSDPUlyssesShardingManager
from verl.models.transformers.patch import llama_forward
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis, rearrange

from transformers import LlamaConfig, MistralConfig, GemmaConfig, Qwen2Config
from transformers import LlamaModel
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForSequenceClassification
# TODO(sgm): add more models for test
# we only need one scale for each model
test_configs = [
    LlamaConfig(num_hidden_layers=2),
    # MistralConfig(num_hidden_layers=1),
    # GemmaConfig(num_hidden_layers=1),
    # Qwen2Config(num_hidden_layers=1)
]


def test_hf_casual_models():
    assert torch.cuda.device_count() >= 4, "need at least 2 gpus for test"
    local_rank, rank, world_size = initialize_global_process_group()
    sp_size = 2
    dp_size = 4
    ulysses_device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(dp_size, sp_size), mesh_dim_names=('dp', 'sp'))
    sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)

    batch_size = 4
    seqlen = 128
    response_length = 127

    for config in test_configs:
        # patch before load
        LlamaModel.forward = llama_forward
        with torch.device('cuda'):
            model = AutoModelForCausalLM.from_config(config=config,
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

        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                   attention_mask)  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

        # unpad the position_ids to align the rotary
        position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                              indices).transpose(0, 1)
        
        # slice input tensor for ulysses
        # input_ids are padded and sliced
        # postition_ids are only padded but not sliced
        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        with sharding_manager:
            logits_split_in_seq = model(input_ids_rmpad, position_ids=position_ids_rmpad,
                                use_cache=False).logits  # (1, total_nnz/n, vocab_size)

        # all_gather output
        logits_full = gather_outpus_and_unpad(logits_split_in_seq, gather_dim=1, unpad_dim=1, padding_size=pad_size)


    print(f'Check pass')

if __name__ == '__main__':
    test_hf_casual_models()
