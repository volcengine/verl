# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from typing import Callable, Optional

from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import make_viewless_tensor
from torch import Tensor

from verl.models.mcore.util import preprocess_packed_seqs
from verl.utils.megatron_utils import unwrap_model

from .util import postprocess_packed_seqs


def gptmodel_forward_1f1b_overlap(
    model: GPTModel,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    labels_mask: Tensor,
    sequence_parallel: bool = False,
    multi_modal_inputs: Optional[dict] = None,
    logits_processor: Optional[Callable] = None,
    logits_processor_args: Optional[dict] = None,
    temperature: float = 1.0,
) -> TransformerModelChunkSchedulePlan:
    assert logits_processor is not None, "fused kernel for 1f1b overlap is not supported yet"
    pre_process: bool = unwrap_model(model).pre_process
    post_process: bool = unwrap_model(model).post_process

    batch_size, seq_len = attention_mask.shape[:2]
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
    input_ids_rmpad = input_ids_rmpad.contiguous()
    labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
    labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
    labels_rmpad = labels_rmpad.contiguous()
    labels_mask_rmpad = labels_mask_rmpad.contiguous()

    schedule_plan = model.build_schedule_plan(
        input_ids=input_ids_rmpad,
        attention_mask=None,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
    )
    if post_process:

        def _custom_post_process_node_forward_impl(self, hidden_states):
            if self.gpt_model.decoder.final_layernorm and not self.gpt_model.mtp_process:
                hidden_states = self.gpt_model.decoder.final_layernorm(hidden_states)
                # TENorm produces a "viewed" tensor. This will result in schedule.py's
                # deallocate_output_tensor() throwing an error, so a viewless tensor is
                # created to prevent this.
                hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

            # Run GPTModel._postprocess
            output_orig = self.gpt_model._postprocess(
                hidden_states=hidden_states,
                input_ids=self.chunk_state.input_ids,
                position_ids=self.chunk_state.position_ids,
                labels=self.chunk_state.labels,
                decoder_input=self.chunk_state.decoder_input,
                rotary_pos_emb=self.chunk_state.rotary_pos_emb,
                rotary_pos_cos=self.chunk_state.rotary_pos_cos,
                rotary_pos_sin=self.chunk_state.rotary_pos_sin,
                mtp_in_postprocess=False,
                loss_mask=self.chunk_state.loss_mask,
                attention_mask=self.chunk_state.attention_mask,
                packed_seq_params=self.chunk_state.packed_seq_params,
                sequence_len_offset=self.chunk_state.sequence_len_offset,
                runtime_gather_output=self.chunk_state.runtime_gather_output,
                extra_block_kwargs=self.chunk_state.extra_block_kwargs,
            )
            if logits_processor:
                args = {
                    k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0]
                    for k, v in logits_processor_args.items()
                }
                output_dict = logits_processor(output_orig, **args)
                output = {
                    k: postprocess_packed_seqs(
                        v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                    )
                    for k, v in output_dict.items()
                }
            else:
                # use fused kernels
                raise NotImplementedError("fused kernel for 1f1b overlap is not supported yet")
            return output

        schedule_plan.post_process.forward_impl = _custom_post_process_node_forward_impl.__get__(
            schedule_plan.post_process, schedule_plan.post_process.__class__
        )

    return schedule_plan
