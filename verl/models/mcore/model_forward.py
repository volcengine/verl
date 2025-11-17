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

from verl.utils.megatron_utils import unwrap_model

from .util import (
    postprocess_packed_seqs,
    postprocess_packed_seqs_no_padding,
    preprocess_packed_seqs,
    preprocess_packed_seqs_no_padding,
)


def model_forward_gen(vision_model: bool = False):
    def model_forward(
        model,
        input_ids,
        attention_mask,
        position_ids,
        multi_modal_inputs: dict,
        logits_processor=None,
        logits_processor_args: dict = None,
        value_model=False,
    ):
        """Forward pass for models with sequence packing."""
        pre_process = (
            unwrap_model(model).pre_process if not vision_model else True
        )  # vision model always needs pre_process
        post_process = unwrap_model(model).post_process

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)

        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids if not vision_model else None,  # vision models will calculate position_ids
            packed_seq_params=packed_seq_params,
            **model_kwargs,
        )
        if post_process and logits_processor is not None:
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
            output = postprocess_packed_seqs(
                output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
            )
        if value_model and post_process:
            output = output[..., 0]
        return output

    return model_forward


def gptmodel_forward_no_padding(
    model,
    input_ids,
    multi_modal_inputs: dict,
    logits_processor=None,
    logits_processor_args: dict = None,
    value_model=False,
):
    """Default forward pass for GPT models with optional sequence packing."""
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process

    model_kwargs = {}
    if "pixel_values" in multi_modal_inputs:
        model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
    if "image_grid_thw" in multi_modal_inputs:
        model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)

    batch_size = input_ids.shape[0]
    input_ids_rmpad, packed_seq_params = preprocess_packed_seqs_no_padding(input_ids, pre_process=pre_process)
    input_ids_rmpad = input_ids_rmpad.contiguous()
    output_orig = model(
        input_ids=input_ids_rmpad,
        attention_mask=None,
        position_ids=None,
        packed_seq_params=packed_seq_params,
        **model_kwargs,
    )

    if post_process and logits_processor is not None:
        args = {k: preprocess_packed_seqs_no_padding(v, pre_process=True)[0] for k, v in logits_processor_args.items()}
        output_dict = logits_processor(output_orig, **args)
        output = {
            k: postprocess_packed_seqs_no_padding(
                v, packed_seq_params, input_ids, batch_size, post_process=post_process
            )
            for k, v in output_dict.items()
        }
    else:
        output = postprocess_packed_seqs_no_padding(
            output_orig, packed_seq_params, input_ids, batch_size, post_process=post_process
        )

    if value_model and post_process:
        # output = output[..., 0]
        # while using nested tensor, the advanced indexing operation above will result in an error at backward, i.e.
        # ValueError: NestedTensor _nested_select_backward_default(grad_output: t, self: jt_all, dim: any, index: any)
        # so we use `squeeze` to remove the last dimension
        output = output.squeeze(-1)

    return output
