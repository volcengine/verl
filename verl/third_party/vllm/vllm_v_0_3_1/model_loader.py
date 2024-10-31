# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader
"""Utilities for selecting and loading models."""
import contextlib
from typing import Dict, Type, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from megatron.core.tensor_parallel.utils import VocabUtility

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.weight_utils import (get_quant_config, initialize_dummy_weights)

from .config import ModelConfig
from vllm.config import DeviceConfig, LoRAConfig
from .weight_loaders import *
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
from vllm.sequence import SamplerOutput
from typing import Optional
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.sampler import _prune_hidden_states, _apply_logits_processors, _apply_penalties, _apply_top_k_top_p, _apply_min_p, _apply_penalties, _sample, _get_logprobs, _build_sampler_output


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return model_cls
    raise ValueError(f"Model architectures {architectures} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


from vllm.model_executor.layers.linear import *
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.layers.activation import ScaledActivation

__LAYER_WEIGHT_LOADER_REGISTRY__ = {
    ColumnParallelLinear: parallel_weight_loader,
    MergedColumnParallelLinear: parallel_weight_loader,
    QKVParallelLinear: parallel_weight_loader,
    RowParallelLinear: parallel_weight_loader,
    VocabParallelEmbedding: parallel_weight_loader,
    ParallelLMHead: parallel_weight_loader
    # "ScaledActivation.weight_loader": ScaledActivation, # TODO(shengguangming): latest commit in vllm fix awq for this function and add load_weights
    # "default_weight_loader": default_weight_loader
}

# NOTE(gmsheng): change the weight_loader function in runtime
for layer_class, weight_loader in __LAYER_WEIGHT_LOADER_REGISTRY__.items():
    layer_class.weight_loader = weight_loader

__MODEL_WEIGHT_LOADER_REGISTRY__ = {
    'GPT2LMHeadModel': gpt2_weight_loader,
    'LlamaForCausalLM': llama_weight_loader,
    'LLaMAForCausalLM': llama_weight_loader,
    'MistralForCausalLM': mistral_weight_loader,
}

# FIXME(shengguangming): the vLLM vocab will pad to 64, which may incur out of bounds
# so we need to rewrite the init function of vocab
DEFAULT_VOCAB_PADDING_SIZE = 64


def vocab_init(self,
               num_embeddings: int,
               embedding_dim: int,
               params_dtype: Optional[torch.dtype] = None,
               org_num_embeddings: Optional[int] = None,
               padding_size: int = DEFAULT_VOCAB_PADDING_SIZE):
    super(VocabParallelEmbedding, self).__init__()

    # Keep the input dimensions.
    # TODO (pad to be divided by 4)
    self.num_embeddings = num_embeddings
    self.org_vocab_size = org_num_embeddings or num_embeddings

    # self.num_embeddings_padded = pad_vocab_size(num_embeddings,
    #                                             padding_size)
    self.embedding_dim = embedding_dim
    if params_dtype is None:
        params_dtype = torch.get_default_dtype()
    self.tp_size = get_tensor_model_parallel_world_size()
    # Divide the weight matrix along the vocaburaly dimension.

    self.vocab_start_index, self.vocab_end_index = (VocabUtility.vocab_range_from_global_vocab_size(
        self.num_embeddings, get_tensor_model_parallel_rank(), self.tp_size))
    self.num_embeddings_per_partition = (self.vocab_end_index - self.vocab_start_index)
    self.weight = Parameter(
        torch.empty(
            self.num_embeddings_per_partition,
            self.embedding_dim,
            # device=torch.cuda.current_device(),
            dtype=params_dtype))
    set_weight_attrs(self.weight, {"parallel_dim": 0, "weight_loader": self.weight_loader})


VocabParallelEmbedding.__init__ = vocab_init


def _get_model_weight_loader(arch: str):
    if arch in __MODEL_WEIGHT_LOADER_REGISTRY__:
        return __MODEL_WEIGHT_LOADER_REGISTRY__[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     f"Supported architectures: {ModelRegistry.get_supported_archs()}")


def get_model(actor_model: Union[PreTrainedModel, Dict],
              model_config: ModelConfig,
              device_config: DeviceConfig,
              lora_config: Optional[LoRAConfig] = None) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)

    # Get the quantization config.
    linear_method = None
    quant_config = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config.quantization, model_config.model, model_config.hf_config,
                                        model_config.download_dir)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(f"The quantization method {model_config.quantization} is not "
                             "supported for the current GPU. "
                             f"Minimum capability: {quant_config.get_min_capability()}. "
                             f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(f"{model_config.dtype} is not supported for quantization "
                             f"method {model_config.quantization}. Supported dtypes: "
                             f"{supported_dtypes}")
        linear_method = quant_config.get_linear_method()

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        # with torch.device(device_config.device):
        # NOTE(sgm): init the model in cpu
        model = model_class(model_config.hf_config, linear_method)

        if model_config.load_format == "dummy":
            model = model.cuda()
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        elif model_config.load_format == 'model' or model_config.load_format == 'auto':
            # NOTE(shengguangming) Load the weights from the actor model
            if isinstance(actor_model, nn.Module):
                load_weights(actor_weights=dict(actor_model.named_parameters(remove_duplicate=False)), vllm_model=model)
            else:
                load_weights(actor_weights=actor_model, vllm_model=model)

        # NOTE(sgm) Some weights are point to gpu, but still need this.
        model = model.cuda()  # NOTE (zhangchi.usc1992) We need this for vllm to profile memory usage
    return model.eval()


# the actor model is .state_dict()
def load_weights(actor_weights: Dict, vllm_model: nn.Module):
    weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    weight_loader(actor_weights, vllm_model)
    # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
    # after init, and we need this after sync model weights for in first iter.
    vllm_model = vllm_model.cuda()


# FIXME(sgm): hack the Sampler function in vllm v0.3.1
# as they use ray, the sampler result will only need to return to the driver node,
# therefore gather is enough. However, we use SPMD instead of a central scheduler,
# all_gather is required (aligned with v0.2.6)
def _get_logits(self, hidden_states: torch.Tensor, embedding: torch.Tensor,
                embedding_bias: Optional[torch.Tensor]) -> torch.Tensor:
    # Get the logits for the next tokens.
    logits = torch.matmul(hidden_states, embedding.t())
    if embedding_bias is not None:
        logits += embedding_bias
    logits = tensor_model_parallel_all_gather(logits)
    # Remove paddings in vocab (if any).
    if logits is not None:
        logits = logits[:, :self.org_vocab_size]
    return logits


def forward(
    self,
    embedding: torch.Tensor,
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    embedding_bias: Optional[torch.Tensor] = None,
) -> Optional[SamplerOutput]:
    # Get the hidden states that we use for sampling.
    hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)

    # Get the logits for the next tokens.
    logits = self._get_logits(hidden_states, embedding, embedding_bias)
    # save origin logprobs for sampler_output
    origin_logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

    # Only perform sampling in the driver worker.
    # Note: `_get_logits` is still distributed across TP workers because
    # the `embedding` weight is distributed across TP workers.
    # TODO(zhuohan): Change the get_logits part to a separate stage.
    if not sampling_metadata.perform_sampling:
        return None

    assert logits is not None
    _, vocab_size = logits.shape

    # Apply logits processors (if any).
    logits = _apply_logits_processors(logits, sampling_metadata)

    # Prepare sampling tensors with pinned memory to avoid blocking.
    (sampling_tensors, do_penalties, do_top_p_top_k,
     do_min_p) = SamplingTensors.from_sampling_metadata(sampling_metadata, vocab_size, logits.device, logits.dtype)

    # Apply presence and frequency penalties.
    if do_penalties:
        logits = _apply_penalties(logits, sampling_tensors.prompt_tokens, sampling_tensors.output_tokens,
                                  sampling_tensors.presence_penalties, sampling_tensors.frequency_penalties,
                                  sampling_tensors.repetition_penalties)

    # Apply temperature scaling.
    # Use in-place division to avoid creating a new tensor.
    logits.div_(sampling_tensors.temperatures.unsqueeze_(dim=1))

    if do_top_p_top_k:
        logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps, sampling_tensors.top_ks)

    if do_min_p:
        logits = _apply_min_p(logits, sampling_tensors.min_ps)

    # We use float32 for probabilities and log probabilities.
    # Compute the probabilities.
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    # Compute the log probabilities.
    # Use log_softmax to ensure numerical stability.
    logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

    # Sample the next tokens.
    sample_results = _sample(probs, logprobs, sampling_metadata)

    # Get the logprobs query results.
    # prompt_logprobs, sample_logprobs = _get_logprobs(
    #     logprobs, sampling_metadata, sample_results)
    prompt_logprobs, sample_logprobs = _get_logprobs(origin_logprobs, sampling_metadata, sample_results)

    return _build_sampler_output(sample_results, sampling_metadata, prompt_logprobs, sample_logprobs)


from vllm.model_executor.layers.sampler import Sampler

Sampler._get_logits = _get_logits
Sampler.forward = forward
