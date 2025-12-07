import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any, Tuple
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock, 
    Qwen3MoeDecoderLayer, 
    Qwen3MoeModel, 
    Qwen3MoeForCausalLM,
    Qwen3MoePreTrainedModel
)

# 尝试导入需要的类，如果失败则跳过
try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:
    from transformers.modeling_utils import Cache
    DynamicCache = Cache

try:
    from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
except ImportError:
    # 如果导入失败，使用基础输出类型
    from transformers.modeling_outputs import BaseModelOutputWithPast as MoeModelOutputWithPast
    from transformers.modeling_outputs import CausalLMOutputWithPast as MoeCausalLMOutputWithPast

try:
    from transformers.generation.utils import GenerationMixin
except ImportError:
    from transformers import GenerationMixin

try:
    from transformers.modeling_attn_mask_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
except ImportError:
    # 如果导入失败，定义简单的替代函数
    def create_causal_mask(*args, **kwargs):
        return None
    def create_sliding_window_causal_mask(*args, **kwargs):
        return None




from typing import Any

def print_input_shapes(
    input_ids: Any = None,
    attention_mask: Any = None,
    position_ids: Any = None,
    routing_ids: Any = None,
    tag='',
) -> None:
    """
    Prints shapes for the given arguments. Handles:
      - torch.Tensor
      - numpy.ndarray
      - list/tuple/dict containing tensors/arrays
      - None
    """

    def shape_of(x):
        # Lazy imports so the function is standalone
        try:
            import torch
        except Exception:
            torch = None
        try:
            import numpy as np
        except Exception:
            np = None

        if x is None:
            return "None"

        if torch is not None and isinstance(x, torch.Tensor):
            return f"{tuple(x.shape)}"

        if np is not None and isinstance(x, np.ndarray):
            return f"{tuple(x.shape)} (numpy)"

        if isinstance(x, (list, tuple)):
            if not x:
                return "[]"
            parts = []
            for i, xi in enumerate(x):
                if torch is not None and isinstance(xi, torch.Tensor):
                    parts.append(f"{i}:{tuple(xi.shape)}")
                elif np is not None and isinstance(xi, np.ndarray):
                    parts.append(f"{i}:{tuple(xi.shape)}(np)")
                else:
                    parts.append(f"{i}:{type(xi).__name__}")
            return f"[{', '.join(parts)}]"

        if isinstance(x, dict):
            parts = []
            for k, v in x.items():
                if torch is not None and isinstance(v, torch.Tensor):
                    parts.append(f"{k}:{tuple(v.shape)}")
                elif np is not None and isinstance(v, np.ndarray):
                    parts.append(f"{k}:{tuple(v.shape)}(np)")
                else:
                    parts.append(f"{k}:{type(v).__name__}")
            return "{" + ", ".join(parts) + "}"

        return f"{type(x).__name__}"

    print(f"[{tag}] input_ids     :", shape_of(input_ids))
    print(f"[{tag}] attention_mask:", shape_of(attention_mask))
    print(f"[{tag}] position_ids  :", shape_of(position_ids))
    print(f"[{tag}] routing_ids   :", shape_of(routing_ids))





class CustomQwen3MoeSparseMoeBlock:
    @staticmethod
    def forward(self, hidden_states: torch.Tensor, routing_map: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if routing_map is not None:
            # 检查一下routing map的尺寸是否相符
            rp_batch_size, rp_sequence, rp_expert_num = routing_map.shape

            assert rp_batch_size == batch_size, f"[qwen3_routing_model][CustomQwen3MoeSparseMoeBlock.forward] Shape mismatch: routing_map.shape={routing_map.shape} but hidden_states.shape={hidden_states.shape}"
            assert rp_sequence == sequence_length, f"[qwen3_routing_model][CustomQwen3MoeSparseMoeBlock.forward] Shape mismatch: routing_map.shape={routing_map.shape} but hidden_states.shape={hidden_states.shape}"
            assert rp_expert_num == self.top_k, f"[qwen3_routing_model][CustomQwen3MoeSparseMoeBlock.forward] Expert number mismatch: rp_expert_num={rp_expert_num} but self.top_k={self.top_k}"

        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        if routing_map is not None:
            # 复用routing map，直接取相应位置的值
            try:
                selected_experts = routing_map.view(-1, self.top_k)#.long() # TODO cx note: review required
                routing_weights = routing_weights.gather(1, selected_experts)
            except RuntimeError as re:
                raise re
        else:
            # TODO CRITICAL DEBUGGING ONLY
            # routing_weights:  value of [batch * sequence_length, self.top_k]
            # selected_experts: indice of [batch * sequence_length, self.top_k]
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
            
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().squeeze(-1)
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

class CustomQwen3MoeDecoderLayer:
    @staticmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        routing_map: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention - 使用更兼容的方式调用
        # import pdb
        # pdb.set_trace()
        print(f"type(hidden_states)={type(hidden_states)}")
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask.to(dtype=torch.float16) if attention_mask is not None else attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # 处理不同版本的返回值格式
        if isinstance(attn_output, tuple):
            hidden_states = attn_output[0]
            self_attn_weights = attn_output[1] if output_attentions else None
            present_key_value = attn_output[-1] if use_cache else None
        else:
            hidden_states = attn_output
            self_attn_weights = None
            present_key_value = None
            
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP forward with optional routing_map
        if hasattr(self.mlp, 'forward') and 'routing_map' in self.mlp.forward.__code__.co_varnames:
            mlp_output = self.mlp(hidden_states, routing_map=routing_map)
        else:
            mlp_output = self.mlp(hidden_states)

        # For the MoE layers, we need to unpack
        if isinstance(mlp_output, tuple):
            hidden_states, router_logits = mlp_output
        else:
            hidden_states = mlp_output
            router_logits = None
            
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        if router_logits is not None:
            outputs += (router_logits,)

        return outputs

class CustomQwen3MoeModel:
    @staticmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        routing_maps = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if routing_maps is not None:
            routing_maps = routing_maps.permute(2, 0, 1, 3)
        # transfer to [layer, batch, seq_length, expert]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            # import pdb
            # pdb.set_trace()
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 创建因果掩码（简化版本）
        causal_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if routing_maps is None:
            routing_maps = [None] * self.config.num_hidden_layers

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer, routing_map in zip(self.layers, routing_maps):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                routing_map=routing_map,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and len(layer_outputs) > (2 if output_attentions else 1):
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits] if v is not None)
        
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    """
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

class CustomMoeForCausalLM:

    @staticmethod
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # routing_maps: Optional[List[torch.Tensor]] = None,
        # routing_ids: Optional[List[torch.Tensor]] = None, # TODO NOTE cx modified
        routing_ids = None,

        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass for the Custom MoE CausalLM model.
        """

        # print_input_shapes(input_ids, attention_mask, position_ids, routing_ids, tag='[cx_debug][qwen3_routing_model][CustomMoeForCausalLM]')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            # routing_maps=routing_maps,
            routing_maps=routing_ids, # TODO NOTE cx modified
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        aux_loss = None
        if output_router_logits:
            router_logits = outputs.router_logits if return_dict else (outputs[-1] if len(outputs) > 1 else None)
            if router_logits:
                aux_loss = load_balancing_loss_func(
                    router_logits,
                    self.num_experts,
                    self.num_experts_per_tok,
                    attention_mask,
                )
                if labels is not None and aux_loss != 0:
                    loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + (outputs[1:] if isinstance(outputs, tuple) else ())
            if aux_loss is not None:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else None,
            hidden_states=outputs.hidden_states if return_dict else None,
            attentions=outputs.attentions if return_dict else None,
            router_logits=outputs.router_logits if return_dict else None,
        )

# 应用monkey patching - 替换原始类的forward方法
def apply_patches():
    """应用所有的monkey patches"""
    # Qwen3MoeSparseMoeBlock.forward = CustomQwen3MoeSparseMoeBlock.forward
    
    # # 为DecoderLayer创建一个wrapper
    # original_decoder_init = Qwen3MoeDecoderLayer.__init__
    # original_decoder_forward = Qwen3MoeDecoderLayer.forward
    
    # def new_decoder_forward(self, *args, **kwargs):
    #     custom_layer = CustomQwen3MoeDecoderLayer()
    #     custom_layer.__dict__.update(self.__dict__)
    #     return custom_layer.forward(*args, **kwargs)
    
    # Qwen3MoeDecoderLayer.forward = new_decoder_forward
    
    # # 应用其他patches
    # Qwen3MoeModel.forward = CustomQwen3MoeModel.forward
    # Qwen3MoeForCausalLM.forward = CustomMoeForCausalLM.forward
    
    Qwen3MoeSparseMoeBlock.forward = CustomQwen3MoeSparseMoeBlock.forward
    Qwen3MoeDecoderLayer.forward = CustomQwen3MoeDecoderLayer.forward
    Qwen3MoeModel.forward = CustomQwen3MoeModel.forward
    Qwen3MoeForCausalLM.forward = CustomMoeForCausalLM.forward

    print("Monkey patched Qwen3Moe Model for FSDP routing replay.")

# 自动应用patches
apply_patches()