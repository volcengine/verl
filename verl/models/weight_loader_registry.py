# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


def get_weight_loader(arch: str):
    from verl.models.llama.megatron.checkpoint_utils.llama_loader import load_state_dict_to_megatron_llama
    from verl.models.qwen2.megatron.checkpoint_utils.qwen2_loader import load_state_dict_to_megatron_qwen2
    _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY = {
        'LlamaForCausalLM': load_state_dict_to_megatron_llama,
        'Qwen2ForCausalLM': load_state_dict_to_megatron_qwen2,
    }

    if arch in _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY:
        return _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     f"Supported architectures: {_MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY.keys()}")
