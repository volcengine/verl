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

from dataclasses import dataclass
from functools import wraps
from types import FunctionType, MethodType
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Cache,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
)
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor

# Import compatibility wrapper for flash_attn_supports_top_left_mask
from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)


def get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: list[int],
    grid_hs: list[int],
    grid_ws: list[int],
):
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
    t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().long()
    _llm_pos_ids = torch.stack([t_index, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)  # + 1 ) # 12.09 by malinhan
    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids


def get_chunked_index(token_indices: torch.Tensor, tokens_per_chunk: int, remove_index: int) -> list[tuple[int, int]]:
    """
    Splits token index list into chunks based on token value ranges.

    Given a list of token indices, returns a list of (start, end) index tuples representing
    slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

    For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
    - the first chunk contains token values < 1000,
    - the second chunk contains values >= 1000 and < 2000, and so on.

    Parameters:
        token_indices (`torch.Tensor` of shape `(seq_len, )`): A monotonically increasing list of
                            token index values.
        t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).
        remove_index (`int`) An index id to subtract from `token_indices` before chunking

    Returns:
        `list[tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                            and end (exclusive) indices of a chunk in `token_indices`.
    """

    def _iter():
        i, start_idx = 0, 0  # skip bos token
        current_chunk = 1
        while i < len(token_indices):  # skip eos token
            if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
                yield (start_idx, i)
                start_idx = i
                current_chunk += 1
            i += 1
        yield (start_idx, len(token_indices))

    return list(_iter())


def get_rope_index(
    processor: Qwen2_5OmniProcessor,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    audio_seqlens: Optional[torch.LongTensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    reference to https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L187
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
    """
    spatial_merge_size = 2
    image_token_id = processor.image_token
    video_token_id = processor.video_token
    audio_token_id = processor.audio_token
    vision_start_token_id = processor.vision_bos_token
    audio_start_token_id = processor.vision_eos_token
    position_id_per_seconds = 25
    seconds_per_chunk = 2

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_idx, video_idx, audio_idx = 0, 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums, audio_nums = 0, 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = torch.sum(input_ids == audio_start_token_id)
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (
                (vision_tokens == audio_start_token_id).sum()
                if use_audio_in_video
                else (vision_tokens == video_token_id).sum()
            )
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
            multimodal_nums = image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio = input_tokens.index(audio_token_id, st)
                else:
                    ed_audio = len(input_tokens) + 1
                min_ed = min(ed_image, ed_video, ed_audio)
                if min_ed == ed_audio:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + audio_len + eos_len
                    audio_idx += 1
                    remain_audios -= 1

                elif min_ed == ed_image:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                    llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + image_len + eos_len
                    image_idx += 1
                    remain_images -= 1

                elif min_ed == ed_video and not use_audio_in_video:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                    ).long()
                    llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + video_len + eos_len
                    video_idx += 1
                    remain_videos -= 1

                elif min_ed == ed_video and use_audio_in_video:
                    text_len = min_ed - st - 2
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                    audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]

                    t_index = (
                        torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                    ).long()
                    video_llm_pos_ids = get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )

                    t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                    video_chunk_indexes = get_chunked_index(video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                    audio_chunk_indexes = get_chunked_index(audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                    sub_len = 0
                    for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                        video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                        audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                        if video_chunk_index is not None:
                            sub_len += video_chunk_index[1] - video_chunk_index[0]

                            llm_pos_ids_list.append(video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]])
                        if audio_chunk_index is not None:
                            sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                            llm_pos_ids_list.append(audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]])
                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                    audio_idx += 1
                    video_idx += 1
                    remain_videos -= 1
                    remain_audios -= 1

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

        return position_ids, mrope_position_deltas
    else:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas


# MONKEY PATCH

"""
Fused Kernels
"""

FORWARD_BASE_MODEL = Qwen2_5OmniThinkerTextModel.forward


@dataclass
class Qwen2_5OmniThinkerOutputForPPO(Qwen2_5OmniThinkerCausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def forward_with_torch_backend(
    self: Qwen2_5OmniThinkerForConditionalGeneration,
    input_ids: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
    audio_feature_lengths: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_audio_in_video: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    video_second_per_grid: Optional[torch.LongTensor] = None,
    temperature=1.0,
    **kwargs,
) -> Qwen2_5OmniThinkerOutputForPPO:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    if not return_dict:
        return_dict = True
    outputs = FORWARD_BASE_MODEL(
        self.model,
        input_ids=input_ids,
        input_features=input_features,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        feature_attention_mask=feature_attention_mask,
        audio_feature_lengths=audio_feature_lengths,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        rope_deltas=rope_deltas,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        use_audio_in_video=use_audio_in_video,
        cache_position=cache_position,
        video_second_per_grid=video_second_per_grid,
        **kwargs,
    )

    hidden_states = outputs[0]
    if not return_dict:
        raise NotImplementedError("forward_with_torch_backend must return_dict=True")

    target = labels if labels is not None else torch.roll(input_ids, shifts=-1, dims=-1)
    fused_linear = FusedLinearForPPO()
    log_probs, entropy = fused_linear.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=target,
        temperature=temperature,
    )

    return Qwen2_5OmniThinkerOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_with_triton_backend(
    self: Qwen2_5OmniThinkerForConditionalGeneration,
    input_ids: Optional[torch.LongTensor] = None,
    input_features: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
    audio_feature_lengths: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_audio_in_video: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    video_second_per_grid: Optional[torch.LongTensor] = None,
    temperature=1.0,
    **kwargs,
) -> Qwen2_5OmniThinkerOutputForPPO:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    if not return_dict:
        return_dict = True
    outputs = FORWARD_BASE_MODEL(
        self.model,
        input_ids=input_ids,
        input_features=input_features,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        feature_attention_mask=feature_attention_mask,
        audio_feature_lengths=audio_feature_lengths,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        rope_deltas=rope_deltas,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        use_audio_in_video=use_audio_in_video,
        cache_position=cache_position,
        video_second_per_grid=video_second_per_grid,
        **kwargs,
    )

    hidden_states = outputs[0]
    if not return_dict:
        raise NotImplementedError("forward_with_triton_backend must return_dict=True")

    target = labels if labels is not None else torch.roll(input_ids, shifts=-1, dims=-1)
    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        target,
        temperature,
        "none",
    )

    return Qwen2_5OmniThinkerOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


"""
patch general model to use thinker
"""


# following code reference to https://github.com/modelscope/ms-swift/blob/main/swift/llm/model/model/qwen.py#L775
def to_device(data: Any, device: str | torch.device | int) -> Any:
    """Move inputs to a device"""
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device) for k, v in data.items()})
    elif isinstance(data, tuple | list):
        return type(data)(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device)
    else:
        return data


def find_device(data):
    """
    Finds the device on which a nested dict/list/tuple of tensors lies (assuming they are all on the same device).

    Args:
        (nested list/tuple/dictionary of `torch.Tensor`): The data we want to know the device of.
    """
    if isinstance(data, Mapping):
        for obj in data.values():
            device = find_device(obj)
            if device is not None:
                return device
    elif isinstance(data, tuple | list):
        for obj in data:
            device = find_device(obj)
            if device is not None:
                return device
    elif isinstance(data, torch.Tensor):
        return data.device


def deep_getattr(obj, attr: str, default=None):
    attrs = attr.split(".")
    for a in attrs:
        if obj is None:
            break
        if isinstance(obj, dict):
            obj = obj.get(a, default)
        else:
            obj = getattr(obj, a, default)
    return obj


def patch_get_input_embeddings(model, embedding_keys: str):
    def get_input_embeddings(self) -> nn.Module:
        return deep_getattr(model, embedding_keys)

    model.get_input_embeddings = MethodType(get_input_embeddings, model)


def use_submodel_func(model, submodel_name: str, func_list: Optional[list[str]] = None) -> None:
    if func_list is None:
        func_list = ["generate", "get_input_embeddings", "gradient_checkpointing_enable", "forward"]
    submodel = getattr(model, submodel_name)

    def _get_new_func(func_name: str):
        _old_func = getattr(submodel, func_name)
        if not isinstance(_old_func, FunctionType):
            _old_func = _old_func.__func__

        @wraps(_old_func)
        def _new_func(self, *args, **kwargs):
            res = _old_func(submodel, *args, **kwargs)
            if func_name == "forward":
                device = find_device(args)
                if device is None:
                    device = find_device(kwargs)
                if hasattr(res, "logits"):
                    res.logits = to_device(res.logits, device)
                if hasattr(res, "loss"):
                    res.loss = to_device(res.loss, device)
                if isinstance(res, dict) and "last_hidden_state" in res:
                    res["last_hidden_state"] = to_device(res["last_hidden_state"], device)
            return res

        return _new_func

    for key in func_list:
        setattr(model, key, MethodType(_get_new_func(key), model))
        if key == "generate" and model.device != submodel.device:
            submodel.__class__.device = model.device
        if key == "forward" and "generate" in func_list:
            setattr(submodel, key, MethodType(_get_new_func(key), submodel))  # fix device_map


BASE_THINKER_FORWARD = Qwen2_5OmniThinkerForConditionalGeneration.forward


def patch_model_mock_logits_spliting(model: Qwen2_5OmniForConditionalGeneration):
    def split_after_forward(*args, **kwargs):
        outputs = BASE_THINKER_FORWARD(*args, **kwargs)
        world_size = get_ulysses_sequence_parallel_world_size()
        # mock slicing
        if world_size > 1:
            logits_rmpad = outputs.logits
            seq_len = logits_rmpad.size(1)
            select_mask = torch.arange(0, seq_len).unsqueeze(0)
            select_mask_pad, _, _ = ulysses_pad_and_slice_inputs(select_mask, sp_size=world_size)
            selected_logits = logits_rmpad[:, select_mask_pad.squeeze(0), :]
            outputs.logits = selected_logits
        return outputs

    model.thinker.__class__.forward = split_after_forward


def patch_model_for_thinker_using(model: Qwen2_5OmniForConditionalGeneration):
    """
    patch Qwen2_5OmniForConditionalGeneration's forward to its thinker's forward,
    for it's forward doesn't support input_ids
    reference https://github.com/modelscope/ms-swift/blob/main/swift/llm/model/model/qwen.py#L775 for more info
    """
    model.disable_talker()
    base_model = model.model if "AWQ" in model.__class__.__name__ else model
    patch_model_mock_logits_spliting(model)
    use_submodel_func(base_model, "thinker")

    base_model.config.thinker_config.text_config.keys_to_ignore_at_inference += ["hidden_states", "attention_mask"]
    if hasattr(base_model.config, "keys_to_ignore_at_inference"):
        base_model.config.keys_to_ignore_at_inference += ["hidden_states", "attention_mask"]
    else:
        base_model.config.keys_to_ignore_at_inference = ["hidden_states", "attention_mask"]
    base_model.config.talker_config.pad_token_id = None
    patch_get_input_embeddings(base_model.thinker.visual, "patch_embed")
    return model, model.thinker
