# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from verl import DataProto
from verl.utils.model import create_random_mask
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches


def test_seqlen_balancing():
    input_ids = torch.randint(low=0, high=10, size=(20, 100))

    attention_mask = create_random_mask(input_ids=input_ids, max_ratio_of_left_padding=0.1, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.5)
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    dataproto = DataProto.from_single_dict(data)
    micro_batches, _, micro_bsz_idx_lst = rearrange_micro_batches(dataproto.batch, max_token_len=300)
    batch = torch.cat(micro_batches)
    micro_bsz_idx = []
    for idx in micro_bsz_idx_lst:
        micro_bsz_idx.extend(idx)
    reverse_idx_map = get_reverse_idx(micro_bsz_idx)
    reverse_idx_map = torch.tensor(reverse_idx_map)
    new_batch = batch[reverse_idx_map]
    torch.testing.assert_close(new_batch, dataproto.batch)
