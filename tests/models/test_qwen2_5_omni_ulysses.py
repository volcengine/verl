# Copyright 2025 Individual Contributor: TomQunChaoA
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
import copy

import torch
import torch.distributed
from flash_attn.bert_padding import index_first_axis, rearrange, unpad_input
from torch.distributed import init_device_mesh
from transformers.models.qwen2_5_omni import Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.models.transformers.qwen2_5_omni import (
    patch_model_for_thinker_using,
)
from verl.protocol import DataProto
from verl.utils.distributed import initialize_global_process_group
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

"""
run with `torchrun --nproc-per-node=2 tests/models/test_qwen2_5_omni_ulysses.py`
"""


def sync_model_parameters_global(layer):
    # synchronize weights
    for p in layer.parameters():
        torch.distributed.broadcast(tensor=p.data, src=0)


def _hf_casual_fwd_bwd(sp_size=2, dp_size=1):
    assert torch.cuda.device_count() >= 2, "need at least 2 gpus for test"
    # in china mainland, set export HF_ENDPOINT=https://hf-mirror.com
    config = Qwen2_5OmniConfig.from_pretrained("Qwen/Qwen2.5-Omni-3B")
    config._attn_implementation = "eager"
    config.thinker_config._attn_implementation = "eager"
    config.thinker_config.vision_config._attn_implementation = "eager"
    config.thinker_config.audio_config._attn_implementation = "eager"

    ulysses_device_mesh = init_device_mesh(
        device_type="cuda", mesh_shape=(dp_size, sp_size), mesh_dim_names=("dp", "sp")
    )
    sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)
    with sharding_manager:
        world_size = get_ulysses_sequence_parallel_world_size()

    batch_size = 1
    seqlen = 128
    # patch before load
    with torch.device("cuda"):
        model = Qwen2_5OmniForConditionalGeneration(config)
        model_no_sp = Qwen2_5OmniForConditionalGeneration(config)
        apply_monkey_patch(model.thinker, ulysses_sp_size=world_size)
        apply_monkey_patch(model_no_sp.thinker, ulysses_sp_size=1)
        patch_model_for_thinker_using(model)
        patch_model_for_thinker_using(model_no_sp)

        model = model.to(device="cuda")
        model_no_sp = model_no_sp.to(device="cuda")
        sync_model_parameters_global(model)

    # different rank will generate different input_ids following fsdp
    input_ids = torch.randint(low=0, high=151900, size=(batch_size, seqlen), device="cuda")
    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_left_padding=0, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.8
    )
    position_ids = compute_position_id_with_mask(
        attention_mask
    )  # TODO(sgm): we can construct the position_ids_rmpad here

    model_inputs = {
        "input_ids": input_ids.cuda(),
        "attention_mask": attention_mask.cuda(),
        "position_ids": position_ids.int().cuda(),
    }

    model_inputs = DataProto.from_dict(model_inputs)
    # 1. perform ulysses forward
    with sharding_manager:
        model_inputs = sharding_manager.preprocess_data(model_inputs)
        input_ids = model_inputs.batch["input_ids"]
        attention_mask = model_inputs.batch["attention_mask"]
        position_ids = model_inputs.batch["position_ids"]
        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
        # unpad the position_ids to align the rotary
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
            input_ids_rmpad,
            position_ids_rmpad=position_ids_rmpad,
            sp_size=world_size,
        )

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        outputs = model(input_ids_rmpad, position_ids=position_ids_rmpad)  # (1, total_nnz/n, vocab_size)
        logits_split_in_seq = outputs.logits
        # all_gather output
        logits_full = gather_outputs_and_unpad(logits_split_in_seq, gather_dim=1, unpad_dim=1)

    # 2. perform normal forward
    set_ulysses_sequence_parallel_group(None)
    input_ids_full = copy.deepcopy(input_ids_rmpad)
    position_ids_full = copy.deepcopy(position_ids_rmpad)
    # model_no_sp = copy.deepcopy(model)
    logits_rmpad_local = model_no_sp(
        input_ids_full, position_ids=position_ids_full, return_logits=True
    ).logits  # (1, total_nnz, vocab_size)

    mean_local = logits_rmpad_local.mean()
    mean_full = logits_full.mean()

    mean_full.backward()
    mean_local.backward()

    # 3. check the gradients
    grad = model.thinker.model.layers[0].self_attn.q_proj.weight.grad
    grad_full = model_no_sp.thinker.model.layers[0].self_attn.q_proj.weight.grad
    assert torch.allclose(mean_local, mean_full, atol=1e-2, rtol=3e-5)
    assert torch.allclose(grad, grad_full, atol=1e-2, rtol=3e-5)


if __name__ == "__main__":
    if not torch.distributed.is_initialized():
        initialize_global_process_group()
    _hf_casual_fwd_bwd()
