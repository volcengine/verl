from typing import Optional, Tuple, Union

import itertools
import torch

from .flash_attention_utils import flash_attention_forward

from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.glm4v import Glm4vProcessor
from transformers.models.glm4v.modeling_glm4v import (
    Glm4vTextModel,
    Glm4vTextAttention,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)
from typing_extensions import Unpack

def get_rope_index(
    processor: Glm4vProcessor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算 GLM4V 的位置编码索引（padding-free 形式）。
    注意：这里的 input_ids 是单条样本的一维张量（batch 维已去掉）。
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image|>")
    video_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|begin_of_video|>")
    video_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|end_of_video|>")
    
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        position_ids = torch.ones(3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device)  # (3, seqlen)
        image_index, video_index = 0, 0
        video_group_index = 0
        
        input_ids_filtered = input_ids[attention_mask == 1]
        input_tokens = input_ids_filtered.tolist()

        input_token_type = []
        video_check_flg = False
        for token in input_tokens:
            if token == video_start_token_id:
                video_check_flg = True
            elif token == video_end_token_id:
                video_check_flg = False

            if token == image_token_id and not video_check_flg:
                input_token_type.append("image")
            elif token == image_token_id and video_check_flg:
                input_token_type.append("video")
            else:
                input_token_type.append("text")

        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        llm_pos_ids_list = []
        video_frame_num = 1
        
        for modality_type, start_idx, end_idx in input_type_group:
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

            if modality_type == "image":
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                image_index += 1
                video_frame_num = 1

            elif modality_type == "video":
                t, h, w = (
                    video_frame_num,
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t,
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )

                for t_idx in range(llm_grid_t):
                    t_index = torch.tensor(t_idx).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(1, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(1, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                video_group_index += 1

                if video_group_index >= video_grid_thw[video_index][0]:
                    video_index += 1
                    video_group_index = 0

                video_frame_num += 1

            else:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                video_frame_num = 1

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = torch.arange(input_ids.shape[0], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids

### legacy
# copied from qwen2vl
# 兼容旧实现：这段注意力前向基于 Qwen2-VL 的思路做了针对 GLM4V 的适配
def glm4_vl_attn_forward(
    self: Glm4vTextAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 未来版本会改为必填
    **kwargs,
) -> Tuple[torch.Tensor, None, None]:
    bsz, q_len, _ = hidden_states.size()  # q_len = seq_length / sp_size
    query_states = self.q_proj(hidden_states)  # (batch_size, seq_length / sp_size, num_heads * head_size)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    attn_output, _ = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=dropout_rate,
        sliding_window=None,
        position_ids=position_ids[0],  # important: pass position ids
    )  # (batch_size, seq_length, num_head / sp_size, head_size)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None, None

## overwrite glm4v text model forward
def decoder_forward(
    self: "Glm4vTextModel",
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    # 只能二选一：要么传 input_ids，要么传 inputs_embeds（两个都传或都不传都会报错）
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("input_ids 和 inputs_embeds 只能二选一（请只指定其中一个）")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    # 计算 text 维度的 position ids；三路维度顺序为 (t, h, w)
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    # 基于 text_position_ids 构造自回归掩码（下三角），用于多模态解码
    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers:
        layer_outputs = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )

# 绑定 decoder_forward 到 HF 的 Glm4vTextModel.forward
Glm4vTextModel.forward = decoder_forward

# 通用的 VLM 透传 forward：将输入交给 self.model(...)，用于顶层 ForConditionalGeneration
def forward_base_model(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    # 可选的 VLM 输入
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    outputs = self.model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        **kwargs,
    )
    return outputs

# Torch 后端 fused-kernel 前向（PPO/GRPO）
def forward_with_torch_backend(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **loss_kwargs,
):
    from verl.models.transformers.dense_common import CausalLMOutputForPPO
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )

    hidden_states = outputs[0]
    if not return_dict:
        raise NotImplementedError("forward_with_torch_backend has to return_dict")

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

# Triton 后端 fused-kernel 前向（PPO/GRPO）
def forward_with_triton_backend(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **loss_kwargs,
):
    from verl.models.transformers.dense_common import CausalLMOutputForPPO
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )

    hidden_states = outputs[0]
    if not return_dict:
        raise NotImplementedError("forward_with_triton_backend has to return_dict")

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )