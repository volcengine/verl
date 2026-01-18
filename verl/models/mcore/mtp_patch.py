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

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    roll_tensor,
)
from megatron.core.utils import unwrap_model


def _get_patching_model(model: torch.nn.Module):
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        return model

    if not (hasattr(model, "language_model") and isinstance(model.language_model, GPTModel)):
        print(f"Model {model.__class__.__name__} is not a supported for fused forward")
        return None

    return model.language_model


def patch_postprocess(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is not None:
        model._postprocess_backup = model._postprocess
        model._postprocess = _megatron_gptmodel_postprocess.__get__(model, model.__class__)


def unpatch_postprocess(model: torch.nn.Module):
    model = _get_patching_model(model)
    if model is not None:
        model._postprocess = model._postprocess_backup


# copy from https://github.com/NVIDIA/Megatron-LM/blob/23e092f41ec8bc659020e401ddac9576c1cfed7e/megatron/core/models/gpt/gpt_model.py
# patch the postprocess method of GPTModel to support advanced features like MTP, 1f1b overlap, etc.
def _megatron_gptmodel_postprocess(
    self,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    mtp_in_postprocess=None,
    loss_mask=None,
    decoder_input=None,
    attention_mask=None,
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    runtime_gather_output=None,
    extra_block_kwargs=None,
    inference_context=None,
):
    """Postprocesses decoder hidden states to generate logits or compute loss.

    Applies Multi-Token Prediction if enabled, generates output logits through
    the output layer, and computes language model loss when labels are provided.
    """
    in_inference_mode = inference_context is not None and not self.training
    if in_inference_mode:
        assert runtime_gather_output, "Inference must always gather TP logits"

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()
    if mtp_in_postprocess:
        hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            embedding=self.embedding,
            **(extra_block_kwargs or {}),
        )

    if not self.post_process:
        return hidden_states

    # Skip when mtp_num_layers is None or 0
    if self.config.mtp_num_layers:
        mtp_labels = labels.clone()
        hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
        hidden_states = hidden_states_list[0]
        if loss_mask is None:
            # if loss_mask is not provided, use all ones as loss_mask
            loss_mask = torch.ones_like(mtp_labels)
        for mtp_layer_number in range(self.config.mtp_num_layers):
            # Calc loss for the current Multi-Token Prediction (MTP) layers.
            mtp_labels, _ = roll_tensor(
                mtp_labels,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )
            loss_mask, num_tokens = roll_tensor(
                loss_mask,
                shifts=-1,
                dims=-1,
                cp_group=self.cp_group,
                packed_seq_params=packed_seq_params,
            )

            # Compute mtp loss without storing logits to save memory.
            mtp_loss = self.compute_output_layer_and_language_model_loss(
                hidden_states_list[mtp_layer_number + 1],
                labels=mtp_labels,
                weight=self.shared_embedding_or_output_weight(),
                sequence_parallel_enabled=self.output_layer.sequence_parallel,
                column_parallel_linear=self.output_layer,
                col_linear_kwargs={
                    "weight": output_weight,
                    "runtime_gather_output": runtime_gather_output,
                },
            )

            mtp_loss = loss_mask * mtp_loss
            if self.training:
                # TODO(shifangx): remove the use of parallel_state here
                # after moving loss logging to loss_func in pretrain_gpt.py
                MTPLossLoggingHelper.save_loss_to_tracker(
                    torch.sum(mtp_loss) / num_tokens,
                    mtp_layer_number,
                    self.config.mtp_num_layers,
                    avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                )
            mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
            if self.config.calculate_per_token_loss:
                hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss)
            else:
                hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss / num_tokens)
    # sequence_parallel_override = False

    # if in_inference_mode and inference_context.materialize_only_last_token_logits:
    #     if inference_context.is_static_batching():
    #         hidden_states = hidden_states[-1:, :, :]
    #     else:
    #         if self.output_layer.sequence_parallel:
    #             # Perform the sequence parallel gather here instead of after the output layer
    #             # because we need to slice the last token logits from the full view of the
    #             # packed logits across all requests.
    #             hidden_states = gather_from_sequence_parallel_region(
    #                 hidden_states, group=self.pg_collection.tp
    #             )
    #             self.output_layer.sequence_parallel = False
    #             sequence_parallel_override = True

    #         # Reshape [B, 1, H] to [1, B, H] → extract each sample’s true last‐token hidden
    #         # state ([B, H]) → unsqueeze back to [B, 1, H]
    #         # (so that the output layer, which expects S×B×H, receives only the final token)
    #         hidden_states = inference_context.last_token_logits(
    #             hidden_states.squeeze(1).unsqueeze(0)
    #         ).unsqueeze(1)

    # if has_config_logger_enabled(self.config) or labels is None:
    if labels is None:
        logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
    else:
        logits = None

    # # Restore sequence parallel execution to the output layer if necessary.
    # if sequence_parallel_override:
    #     assert (
    #         in_inference_mode
    #         and inference_context.is_dynamic_batching()
    #         and inference_context.materialize_only_last_token_logits
    #     )
    #     self.output_layer.sequence_parallel = True

    # if has_config_logger_enabled(self.config):
    #     payload = OrderedDict(
    #         {
    #             'input_ids': input_ids,
    #             'position_ids': position_ids,
    #             'attention_mask': attention_mask,
    #             'decoder_input': decoder_input,
    #             'logits': logits,
    #         }
    #     )
    #     log_config_to_disk(self.config, payload, prefix='input_and_logits')

    if labels is None:
        # [s b h] => [b s h]
        return logits.transpose(0, 1).contiguous()

    loss = self.compute_output_layer_and_language_model_loss(
        hidden_states,
        labels=labels,
        weight=self.shared_embedding_or_output_weight(),
        sequence_parallel_enabled=self.output_layer.sequence_parallel,
        column_parallel_linear=self.output_layer,
        col_linear_kwargs={
            "weight": output_weight,
            "runtime_gather_output": runtime_gather_output,
        },
    )

    return loss
