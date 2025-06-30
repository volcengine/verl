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

import warnings
from typing import Union

import torch

from verl.utils.megatron_utils import unwrap_model
from verl.utils.model import CausalLMOutputForPPO

from .util import postprocess_packed_seqs, postprocess_packed_seqs_for_dict_output, preprocess_packed_seqs, recover_left_padding, remove_left_padding


def gptmodel_forward(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    """
    Default forward pass for GPT models with optional sequence packing.

    Args:
        model: The model to run the forward pass on.
        input_ids (torch.Tensor): Input token IDs.
        attention_mask (torch.Tensor): Attention mask for the input.
        position_ids (torch.Tensor): Position IDs for the input tokens.
        sequence_parallel (bool): Whether to use sequence parallelism.
        value_model (bool): Whether the model is a value model.
        pack_seqs (bool): Whether to pack sequences for efficiency.
        * logits_processor (callable, optional): A function to process logits.
        * logits_processor_args (dict, optional): Arguments for the logits processor.
    Returns:
        output: The output of the model after processing.
    """
    pre_process: bool = unwrap_model(model).pre_process
    post_process: bool = unwrap_model(model).post_process

    if pack_seqs:
        # Preprocess packed sequences
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()

        # Main forward pass
        output_orig: Union[torch.Tensor, CausalLMOutputForPPO] = model(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids, labels=None, packed_seq_params=packed_seq_params)
        # output_orig is CausalLMOutputForPPO, hidden_states or directly a logits tensor

        # Post-processing
        if post_process and logits_processor is not None:
            # use logits_processor to calculate smaller data, output_orig is a logits tensor
            args = {k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0] for k, v in logits_processor_args.items()}
            output_dict = logits_processor(output_orig, **args)
            output = {k: postprocess_packed_seqs(v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process) for k, v in output_dict.items()}
        else:
            if logits_processor is None:
                warnings.warn("logits_processor is not provided, may be memory in-efficient", stacklevel=2)
            # default, without any optimization, output_orig is a logits tensor
            output = postprocess_packed_seqs(output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
    else:
        """_summary_
        
        Non-packed sequence forward pass. This is less efficient and should be avoided if possible.
        """
        warnings.warn("Non-packed sequence can have a severe performance penalty, consider using pack_seqs=True.", stacklevel=2)
        assert logits_processor is None, "logits_processor is not supported for non-packed sequence"

        # preprocess, remove left padding
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process)

        # Main forward pass
        output = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids)

        # Post-process, recover left padding
        output = recover_left_padding(output, new_attention_mask, attention_mask, sequence_length, post_process=post_process)

    if value_model and post_process:
        output = output[..., 0]
    return output


def gptmodel_forward_with_fused_kernel(
    model,
    input_ids,
    attention_mask,
    position_ids,
    labels,
    labels_mask,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    **kwargs,
):
    """
    Default forward pass for GPT models with optional sequence packing.

    Args:
        model: The model to run the forward pass on.
        input_ids (torch.Tensor): Input token IDs.
        attention_mask (torch.Tensor): Attention mask for the input.
        position_ids (torch.Tensor): Position IDs for the input tokens.
        * labels (torch.Tensor, optional): Labels for the input, required.
        * labels_mask (torch.Tensor, optional): labels_mask for the input, required.
        sequence_parallel (bool): Whether to use sequence parallelism.
        value_model (bool): Whether the model is a value model.
        pack_seqs (bool): Whether to pack sequences for efficiency.
    Returns:
        output: The output of the model after processing.
    """
    assert labels is not None and labels_mask is not None, "labels and labels_mask must be provided when use fused kernels"
    pre_process: bool = unwrap_model(model).pre_process
    post_process: bool = unwrap_model(model).post_process

    if pack_seqs:
        # Preprocess packed sequences
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        labels_rmpad = None
        if labels is not None:
            # Fused kernels requirements
            labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
            labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
            labels_rmpad = labels_rmpad.contiguous()
            labels_mask_rmpad = labels_mask_rmpad.contiguous()

        # Main forward pass
        output_orig: Union[torch.Tensor, CausalLMOutputForPPO] = model(input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids, labels=labels_rmpad, packed_seq_params=packed_seq_params)
        # output_orig is CausalLMOutputForPPO, hidden_states or directly a logits tensor

        # Post-processing
        if post_process:
            # output_orig is in type of CausalLMOutputForPPO
            output = postprocess_packed_seqs_for_dict_output(labels_mask_rmpad, output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
        else:
            # default, without any optimization, output_orig is a logits tensor
            output = postprocess_packed_seqs(output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
    else:
        """_summary_
        
        Non-packed sequence forward pass. This is less efficient and should be avoided if possible.
        """
        warnings.warn("Non-packed sequence can have a severe performance penalty, consider using pack_seqs=True.", stacklevel=2)

        # preprocess, remove left padding
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process)

        # Main forward pass
        output = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids)

        # Post-process, recover left padding
        output = recover_left_padding(output, new_attention_mask, attention_mask, sequence_length, post_process=post_process)

    if value_model and post_process:
        output = output[..., 0]
    return output


def gptmodel_forward_qwen2_5_vl(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    multi_modal_inputs=None,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    """_summary_
    Forward pass for Qwen2.5 VL models with optional sequence packing and multi-modal inputs.

    Args:
        model: The model to run the forward pass on.
        input_ids (torch.Tensor): Input token IDs.
        attention_mask (torch.Tensor): Attention mask for the input.
        position_ids (torch.Tensor): Position IDs for the input tokens.
        sequence_parallel (bool): Whether to use sequence parallelism.
        value_model (bool): Whether the model is a value model.
        pack_seqs (bool): Whether to pack sequences for efficiency.
        multi_modal_inputs (dict, optional): Multi-modal inputs for the model, such as pixel values and image grid dimensions.
        * logits_processor (callable, optional): A function to process logits.
        * logits_processor_args (dict, optional): Arguments for the logits processor.
    Returns:
        output: The output of the model after processing.
    """
    from megatron.core import parallel_state as mpu

    assert mpu.get_context_parallel_world_size() == 1, "qwen2_5_vl's context parallel is not accurate yet"
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    pixel_values = multi_modal_inputs["pixel_values"].to(input_ids.device) if "pixel_values" in multi_modal_inputs else None
    image_grid_thw = multi_modal_inputs["image_grid_thw"].to(input_ids.device) if "image_grid_thw" in multi_modal_inputs else None
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        if post_process and logits_processor is not None:
            args = {k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0] for k, v in logits_processor_args.items()}
            output_dict = logits_processor(output_orig, **args)
            output = {k: postprocess_packed_seqs(v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process) for k, v in output_dict.items()}
        else:
            if logits_processor is None:
                warnings.warn("logits_processor is not provided, may be memory in-efficient", stacklevel=2)
            output = postprocess_packed_seqs(output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
    else:
        warnings.warn("Non-packed sequence can have a severe performance penalty, consider using pack_seqs=True.", stacklevel=2)
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process)
        output = model(
            input_ids=new_input_ids,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        output = recover_left_padding(output, new_attention_mask, attention_mask, sequence_length, post_process=post_process)
    if value_model and post_process:
        output = output[..., 0]
    return output


def gptmodel_forward_qwen2_5_vl_with_fused_kernel(
    model,
    input_ids,
    attention_mask,
    position_ids,
    labels,
    labels_mask,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    multi_modal_inputs=None,
    **kwargs,
):
    """_summary_
    Forward pass for Qwen2.5 VL models with optional sequence packing and multi-modal inputs, using fused kernels.

    Args:
        model: The model to run the forward pass on.
        input_ids (torch.Tensor): Input token IDs.
        attention_mask (torch.Tensor): Attention mask for the input.
        position_ids (torch.Tensor): Position IDs for the input tokens.
        * labels (torch.Tensor, optional): Labels for the input, required if use used kernels.
        * labels_mask (torch.Tensor, optional): Labels mask for the input, required if use used kernels.
        sequence_parallel (bool): Whether to use sequence parallelism.
        value_model (bool): Whether the model is a value model.
        pack_seqs (bool): Whether to pack sequences for efficiency.
        multi_modal_inputs (dict, optional): Multi-modal inputs for the model, such as pixel values and image grid dimensions.
    """
    from megatron.core import parallel_state as mpu

    assert labels is not None and labels_mask is not None, "labels and labels_mask must be provided when use fused kernels"
    assert mpu.get_context_parallel_world_size() == 1, "qwen2_5_vl's context parallel is not accurate yet"
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    pixel_values = multi_modal_inputs["pixel_values"].to(input_ids.device) if "pixel_values" in multi_modal_inputs else None
    image_grid_thw = multi_modal_inputs["image_grid_thw"].to(input_ids.device) if "image_grid_thw" in multi_modal_inputs else None
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        labels_rmpad = None
        if labels is not None:
            # Fused kernels requirements
            labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
            labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
            labels_rmpad = labels_rmpad.contiguous()
            labels_mask_rmpad = labels_mask_rmpad.contiguous()
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            labels=labels_rmpad,
            packed_seq_params=packed_seq_params,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # Post-processing
        if post_process:
            # output_orig is in type of CausalLMOutputForPPO
            output = postprocess_packed_seqs_for_dict_output(labels_mask_rmpad, output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
        else:
            # default, without any optimization, output_orig is a logits tensor
            output = postprocess_packed_seqs(output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process)
    else:
        warnings.warn("Non-packed sequence can have a severe performance penalty, consider using pack_seqs=True.", stacklevel=2)
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process)
        output = model(
            input_ids=new_input_ids,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        output = recover_left_padding(output, new_attention_mask, attention_mask, sequence_length, post_process=post_process)
    if value_model and post_process:
        output = output[..., 0]
    return output
