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

from collections import OrderedDict
from typing import Optional

import torch
from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from torch import Tensor

from verl.models.mcore.util import preprocess_packed_seqs
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.megatron_utils import unwrap_model
from verl.utils.model import CausalLMOutputForPPO

from .qwen2_5_vl.model import Qwen2_5VLModel
from .util import postprocess_packed_seqs_for_dict_output


def patch_fused_forward(model: torch.nn.Module):
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        model = model
    elif isinstance(model, Qwen2_5VLModel):
        if not hasattr(model, "language_model"):
            # the qwen2.5vl model might only have vision_model
            return
        model = model.language_model
    else:
        raise ValueError("Model is not a GPTModel or Qwen2_5VLModel")
    model.forward_backup = model.forward
    model.forward = _fused_GPTModel_forward.__get__(model, model.__class__)
    return


def unpatch_fused_forward(model: torch.nn.Module):
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        model = model
    elif isinstance(model, Qwen2_5VLModel):
        model = model.language_model
    else:
        raise ValueError("Model is not a GPTModel or Qwen2_5VLModel")
    model.forward = model.forward_backup
    return


def fused_forward_gptmodel(
    model: GPTModel,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    labels_mask: Tensor,
    **kwargs,
):
    pre_process: bool = unwrap_model(model).pre_process
    post_process: bool = unwrap_model(model).post_process

    batch_size, seq_len = attention_mask.shape[:2]
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
    input_ids_rmpad = input_ids_rmpad.contiguous()
    labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
    labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
    labels_rmpad = labels_rmpad.contiguous()
    labels_mask_rmpad = labels_mask_rmpad.contiguous()

    output_orig: CausalLMOutputForPPO = model(
        input_ids=input_ids_rmpad,
        attention_mask=None,
        position_ids=position_ids,
        labels=labels_rmpad,
        packed_seq_params=packed_seq_params,
    )

    if post_process:
        # output_orig is in type of CausalLMOutputForPPO
        output = postprocess_packed_seqs_for_dict_output(
            labels_mask_rmpad,
            output_orig,
            packed_seq_params,
            attention_mask,
            batch_size,
            seq_len,
            post_process=post_process,
        )
    else:
        output = output_orig
    return output


def fused_forward_qwen2_5_vl(
    model: Qwen2_5VLModel,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    labels_mask: Tensor,
    multi_modal_inputs=None,
    **kwargs,
):
    # pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process

    pixel_values = (
        multi_modal_inputs["pixel_values"].to(input_ids.device) if "pixel_values" in multi_modal_inputs else None
    )
    image_grid_thw = (
        multi_modal_inputs["image_grid_thw"].to(input_ids.device) if "image_grid_thw" in multi_modal_inputs else None
    )

    batch_size, seq_len = attention_mask.shape[:2]
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
    labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
    labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
    labels_rmpad = labels_rmpad.contiguous()
    labels_mask_rmpad = labels_mask_rmpad.contiguous()
    input_ids_rmpad = input_ids_rmpad.contiguous()
    output_orig: CausalLMOutputForPPO = model(
        input_ids=input_ids_rmpad,
        attention_mask=None,
        position_ids=position_ids,
        packed_seq_params=packed_seq_params,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        labels=labels_rmpad,
    )
    if post_process:
        # output_orig is in type of CausalLMOutputForPPO
        output = postprocess_packed_seqs_for_dict_output(
            labels_mask_rmpad,
            output_orig,
            packed_seq_params,
            attention_mask,
            batch_size,
            seq_len,
            post_process=post_process,
        )
    else:
        output = output_orig
    return output


def _fused_GPTModel_forward(
    self,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context: BaseInferenceContext = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
    runtime_gather_output: Optional[bool] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
    loss_mask: Optional[Tensor] = None,
    temperature: float = 1.0,
) -> CausalLMOutputForPPO:
    """
    Forward pass for GPT models with fused kernel support.

    Patch https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py
    """

    # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
    # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

    # Decoder embedding.
    if decoder_input is not None:
        pass
    elif self.pre_process:
        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
    else:
        # intermediate stage of pipeline
        # decoder will get hidden_states from encoder.input_tensor
        decoder_input = None

    # Rotary positional embeddings (embedding is None for PP intermediate devices)
    rotary_pos_emb = None
    rotary_pos_cos = None
    rotary_pos_sin = None
    if self.position_embedding_type == "rope" and not self.config.multi_latent_attention:
        if not self.training and self.config.flash_decode and inference_context:
            assert inference_context.is_static_batching(), "GPTModel currently only supports static inference batching."
            # Flash decoding uses precomputed cos and sin for RoPE
            rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                inference_context.max_sequence_length,
                self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
            )
        else:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_context, self.decoder, decoder_input, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == "thd",
            )
    elif self.position_embedding_type == "mrope" and not self.config.multi_latent_attention:
        if self.training or not self.config.flash_decode:
            rotary_pos_emb = self.rotary_pos_emb(position_ids, self.mrope_section)
        else:
            # Flash decoding uses precomputed cos and sin for RoPE
            raise NotImplementedError(
                "Flash decoding uses precomputed cos and sin for RoPE, not implmented in MultimodalRotaryEmbedding yet."
            )

    if (
        (self.config.enable_cuda_graph or self.config.flash_decode)
        and rotary_pos_cos is not None
        and inference_context
        and inference_context.is_static_batching()
        and not self.training
    ):
        sequence_len_offset = torch.tensor(
            [inference_context.sequence_len_offset] * inference_context.current_batch_size,
            dtype=torch.int32,
            device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
        )
    else:
        sequence_len_offset = None

    # Wrap decoder_input to allow the decoder (TransformerBlock) to delete the
    # reference held by this caller function, enabling early garbage collection for
    # skip inference

    # Run decoder.
    hidden_states = self.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **(extra_block_kwargs or {}),
    )

    # Process inference output.
    if inference_context and not inference_context.is_static_batching():
        hidden_states = inference_context.last_token_logits(hidden_states.squeeze(1).unsqueeze(0)).unsqueeze(1)

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if self.mtp_process:
        hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            loss_mask=loss_mask,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            embedding=self.embedding,
            output_layer=self.output_layer,
            output_weight=output_weight,
            runtime_gather_output=runtime_gather_output,
            compute_language_model_loss=self.compute_language_model_loss,
            **(extra_block_kwargs or {}),
        )

    if not self.post_process:
        return hidden_states

    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    if self.config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)
    logprobs, entropy = linear_cross_entropy(
        hidden_states,
        self.output_layer.weight,
        labels,
        temperature,
        "none",
        parallel_state.get_tensor_model_parallel_group(),
    )

    if has_config_logger_enabled(self.config):
        payload = OrderedDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(self.config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs

    return output
