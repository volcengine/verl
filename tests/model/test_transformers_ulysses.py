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
from dist_attn.ulysses.ops import gather_outputs
from dist_attn.ulysses.parallel_states import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from verl.utils.ulysses import ulysses_pad_and_slice_inputs
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

def sync_model_parameters_global(layer):
    # synchronize weights
    for p in layer.parameters():
        torch.distributed.broadcast(tensor=p.data, src=0)


def test_hf_casual_models():
    assert torch.cuda.device_count() >= 2, "need at least 2 gpus for test"
    local_rank, rank, world_size = initialize_global_process_group()
    sp_size = 8
    dp_size = 1
    ulysses_device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(dp_size, sp_size), mesh_dim_names=('dp', 'sp'))
    sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)

    batch_size = 1
    seqlen = 2
    response_length = 127

    for config in test_configs:
        # patch before load
        LlamaFlashAttention2.forward = llama_flash_attn_forward
        with torch.device('cuda'):
            model = AutoModelForCausalLM.from_config(config=config,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation='flash_attention_2')
            model = model.to(device='cuda')
            sync_model_parameters_global(model)
        
        # different rank will generate different input_ids following fsdp
        input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device='cuda')
        attention_mask = create_random_mask(input_ids=input_ids,
                                            max_ratio_of_left_padding=0,
                                            max_ratio_of_valid_token=1,
                                            min_ratio_of_valid_token=1)
        position_ids = compute_position_id_with_mask(
            attention_mask)  # TODO(sgm): we can construct the position_ids_rmpad here

        model_inputs = {
            'input_ids': input_ids.cuda(),
            'attention_mask': attention_mask.cuda(),
            'position_ids': position_ids.int().cuda()
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
            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                indices).transpose(0, 1)

            # slice input tensor for ulysses
            # input_ids are padded and sliced
            # postition_ids are only padded but not sliced
            input_ids_rmpad_sliced, _, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
            print(f'rank: {torch.distributed.get_rank()} input_ids after slice: {input_ids_rmpad_sliced}')
            print(f'pad size: {pad_size}')

            # input with input_ids_rmpad and postition_ids to enable flash attention varlen
            logits_split_in_seq = model(input_ids_rmpad_sliced, position_ids=position_ids_rmpad,
                                        use_cache=False).logits  # (1, total_nnz/n, vocab_size)
            
            # all_gather output
            logits_full = gather_outputs(logits_split_in_seq, 1)

        # 2. perform normal forward
        set_ulysses_sequence_parallel_group(None)
        logits_rmpad_local = model(input_ids_rmpad, position_ids=position_ids_rmpad,
                             use_cache=False).logits  # (1, total_nnz, vocab_size)
        
        if torch.distributed.get_rank() == 7:
            print(f'logits_rmpad_local: {logits_rmpad_local[0, :, :4]}')
        print(f'rank: {torch.distributed.get_rank()}: logits_split_in_seq: {logits_split_in_seq[0, :, :4]}')
        mean_local = logits_rmpad_local.mean()
        mean_full = logits_full.mean()
        torch.testing.assert_close(mean_local, mean_full, rtol=1e-2, atol=1e-5)
        torch.testing.assert_close(logits_rmpad_local, logits_full[:, :logits_rmpad_local.shape[1]], rtol=1e-1, atol=1e-5)
    print(f'Check pass')

if __name__ == '__main__':
    test_hf_casual_models()