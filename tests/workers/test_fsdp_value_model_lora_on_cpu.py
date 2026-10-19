# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""LoRA on the PPO critic (AutoModelForTokenClassification) must not use the CausalLM PEFT wrapper."""

from types import SimpleNamespace

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2ForTokenClassification

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


def _tiny_config(**kwargs):
    return Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_labels=1,
        **kwargs,
    )


def _build(module, model_type, exclude_modules=None):
    model_config = SimpleNamespace(
        model_type=model_type,
        lora_adapter_path=None,
        lora_rank=4,
        lora_alpha=8,
        target_modules="all-linear",
        target_parameters=None,
        exclude_modules=exclude_modules,
    )
    engine = SimpleNamespace(model_config=model_config)
    return FSDPEngine._build_lora_module(engine, module)


def test_value_model_lora_builds_and_trains_head_fully():
    module = Qwen2ForTokenClassification(_tiny_config())
    peft_model = _build(module, "value_model")

    assert peft_model.peft_config["default"].task_type is None
    names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    # LoRA adapters on the backbone, head trained in full via modules_to_save, no LoRA on the head.
    assert any("lora_A" in n for n in names)
    assert not any("score" in n and "lora_" in n for n in names)
    assert any("score.modules_to_save" in n for n in names)

    out = peft_model(input_ids=torch.randint(0, 64, (2, 5)))
    out.logits.sum().backward()


def test_language_model_lora_keeps_causal_lm_task_type():
    peft_model = _build(Qwen2ForCausalLM(_tiny_config()), "language_model")
    assert getattr(peft_model.peft_config["default"].task_type, "value", None) == "CAUSAL_LM"


def test_value_model_lora_string_exclude_modules_stays_a_regex():
    module = Qwen2ForTokenClassification(_tiny_config())
    peft_model = _build(module, "value_model", exclude_modules=r".*\.q_proj")
    names = [n for n, _ in peft_model.named_parameters()]
    assert not any("q_proj.lora_" in n for n in names)  # user's regex still excludes q_proj
    assert any("k_proj.lora_A" in n for n in names)
    assert not any("score" in n and "lora_" in n for n in names)  # head still excluded


def test_value_model_head_survives_lora_only_save_filter_and_hf_export_names():
    from verl.utils.fsdp_utils import normalize_peft_param_name

    peft_model = _build(Qwen2ForTokenClassification(_tiny_config()), "value_model")
    state = peft_model.state_dict()
    assert any(".modules_to_save." in k for k in state)  # what save_lora_only must keep
    exported = normalize_peft_param_name(state)
    assert "score.weight" in exported and "score.bias" in exported
    assert not any("original_module" in k or "modules_to_save" in k for k in exported)
