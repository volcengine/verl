# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Alibaba PAI Team.
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


import logging

import torch
from megatron.core import InferenceParams, mpu, tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel

# from .transformer_config import Qwen2VLTransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from verl.models.mcore.util import preprocess_packed_seqs

from .attention import Qwen2_5VLSelfAttention
from .vision_model import Qwen2_5VisionModel


# Note: This is under development and may be missing features.
class Qwen2_5VLModel(MegatronModule):
    """Qwen2.5VL multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the
            language model.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length. This is used for
            positional embedding.
        vision_transformer_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the
            vision model.
        vision_projection_config (TransformerConfig): Config for the projection from vision model outputs to
            language model inputs.
        vision_projection_layer_spec (ModuleSpec): Specifies the module to use for the vision
            projection.
        vision_projection_type (str): Type of the vision projection to use. Default is a 2-layer MLP.
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks. This
            is typically True for training and False for inference.
        language_rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings
            in the language model. Defaults to 1.0.
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism).
            Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline
            parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        img_h (int): The height of each image that the ViT will see.
        img_w (int): The width of each image that the ViT will see.
        patch_dim (int): The size of each patch side.
        img_embedding_idx (int): Index in the language_embeddings tensor where image_embeddings should be
            inserted. Defaults to 0.
    """

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        vision_projection_type: str = "mlp",
        parallel_output: bool = True,
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 10000,
        fp16_lm_cross_entropy: bool = False,
        language_share_embeddings_and_output_weights: bool = False,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
    ) -> None:
        super().__init__(config=language_transformer_config)

        # patch self_attention to use qwen2_5_vl attention
        vision_transformer_layer_spec.submodules.self_attention.module = Qwen2_5VLSelfAttention
        for layer_spec in language_transformer_layer_spec.layer_specs:
            layer_spec.submodules.self_attention.module = Qwen2_5VLSelfAttention

        logging.getLogger(__name__).warning("Qwen2VL model is under development and may be missing features.")

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.vision_projection = None
        self.language_model = None
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        self.square_merge_size = vision_projection_config.ffn_hidden_size // vision_transformer_config.hidden_size

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        if self.pre_process:
            self.vision_model = Qwen2_5VisionModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                vision_projection_config,
                vision_projection_layer_spec,
                projection_type=vision_projection_type,
                pre_process=True,
                post_process=True,
            )

        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_rotary_base,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
        )
        assert mpu.get_context_parallel_world_size() <= 1, "please use mbridge for qwen2_5_vl with context parallelism"
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen2VL"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function of the Qwen2VL model.
        ### there is a workaround for supporting sequence packing with context parallelism
        # cp split with sequence packing will make model lose vision token information, so we need to keep
        # the original input_ids and pack them after vision embedding is calculated,
        # cooporate with verl's models/mcore/model_forward.py
        # pack the combined_embeddings to thd here, we check if packed_seq_params is None to determine if
        #  we need to pack the combined_embeddings to thd
        # this function needs the position_ids and attention_mask in BSHD format, no matter use packed_seq or not

        Args:
            image_data (torch.Tensor): input image of shape [total_thw_size, n_features].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len,
                combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.

            video_start_index:
                0 -- all video
                len(video_seq) -- all image
                others -- mixture
            *_input_mask: should not be None in the first PP stage
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape
                [b, s, vocab_size].
        """
        video_start_index = 0
        vision_grid_thw = None
        vision_data = None
        if image_grid_thw is not None:
            image_mask = input_ids == self.image_token_id
            vision_grid_thw = image_grid_thw
            vision_data = pixel_values
            video_start_index = image_mask.sum().item()
        if video_grid_thw is not None:
            video_mask = input_ids == self.video_token_id
            if vision_grid_thw is not None:
                vision_grid_thw = torch.cat([vision_grid_thw, video_grid_thw], dim=0)
                vision_data = torch.cat([vision_data, pixel_values_videos], dim=0)
            else:
                vision_grid_thw = video_grid_thw
                vision_data = pixel_values_videos
        use_inference_kv_cache = (
            inference_params is not None and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if use_inference_kv_cache:
            raise NotImplementedError()

        if self.pre_process:
            vision_embeds = None
            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                vision_embeds = self.vision_model(
                    vision_data=vision_data,  # If None, vision model should use intermediate outputs (EPP > 1)
                    grid_thw=vision_grid_thw,  # should provided in each EPP stage
                )

            # If running inference, the language model KV cache will be updated for image token positions.
            # Here we store the image tokens sequence length, which can be used as an offset to the KV cache later.
            if inference_params is not None:
                raise NotImplementedError()
                # inference_params.key_value_memory_dict["image_tokens_count"] = (
                #     vision_embeddings.shape[0]
                # )

            # If running inference, we can skip image token computation if they were computed already earlier
            # for this sample.
            if use_inference_kv_cache:
                language_embeddings: torch.Tensor = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None,  # NOTE: disable
                )  # [text_seq_len, b, h_language]
                # NOTE: why not cat here? is it the combined embeddings useless?
                combined_embeddings = language_embeddings
            elif vision_embeds is not None:
                if video_start_index == 0:
                    image_embeds = None
                    video_embeds = vision_embeds
                elif video_start_index == vision_embeds.shape[0]:
                    image_embeds = vision_embeds
                    video_embeds = None
                elif 0 < video_start_index < vision_embeds.shape[0]:
                    image_embeds = vision_embeds[:video_start_index]
                    video_embeds = vision_embeds[video_start_index:]
                else:
                    raise ValueError(
                        f"Expect video token start index in range [0, {vision_embeds.shape[0]}], but got "
                        f"{video_start_index}"
                    )

                combined_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None,  # NOTE: disable
                )  # [text_seq_len, b, h_language]

                if image_embeds is not None or video_embeds is not None:
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                    if image_embeds is not None:
                        image_mask = (input_ids == self.image_token_id).contiguous()
                        if image_mask.sum() > 0:
                            combined_embeddings = combined_embeddings.clone()
                            combined_embeddings[image_mask] = image_embeds.to(
                                dtype=combined_embeddings.dtype, device=combined_embeddings.device
                            )
                    if video_embeds is not None:
                        video_mask = (input_ids == self.video_token_id).contiguous()
                        if video_mask.sum() > 0:
                            combined_embeddings = combined_embeddings.clone()
                            combined_embeddings[video_mask] = video_embeds.to(
                                dtype=combined_embeddings.dtype, device=combined_embeddings.device
                            )
                    combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            else:
                combined_embeddings = self.language_model.embedding(
                    input_ids=input_ids,
                    position_ids=None,  # NOTE: disable
                )  # [text_seq_len, b, h_language]

            if packed_seq_params is not None:
                combined_embeddings = (
                    preprocess_packed_seqs(
                        combined_embeddings.transpose(0, 1).contiguous(), attention_mask, pre_process=True
                    )[0]
                    .transpose(0, 1)
                    .contiguous()
                )
            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None
        from .rope_utils import get_rope_index

        # BSHD
        position_ids, _ = get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        # THD
        if packed_seq_params is not None:
            position_ids = (
                preprocess_packed_seqs(position_ids.permute(1, 2, 0), attention_mask, pre_process=True)[0]
                .permute(2, 0, 1)
                .contiguous()
            )
            attention_mask = None

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            # inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,  # currently always None
            **(extra_block_kwargs or {}),
            **kwargs,
        )

        return output
