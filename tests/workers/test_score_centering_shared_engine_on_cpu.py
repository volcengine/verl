# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Actual actor loss and tiny-Transformer gradient gates for shared FSDP scoring."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo.score_centering import score_centering_correction, score_centering_support_correction
from verl.utils import tensordict_utils as tu
from verl.utils import torch_functional as verl_F
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding


def actor_fixture(device="cpu"):
    prompts = torch.tensor([[4, 8, 3], [0, 5, 9]], device=device)
    responses = torch.tensor([[10, 11, 7, 1, 6], [12, 13, 2, 0, 0]], device=device)
    input_ids = torch.cat((prompts, responses), dim=-1)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0, 0]], device=device)
    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
            "response_mask": mask[:, 3:],
            "advantages": torch.tensor([[0.8, -1.0, 1.2, 0.4, -0.6], [1.1, -0.4, 0.9, 0, 0]], device=device),
            "old_log_probs": torch.full((2, 5), -2.0, device=device),
            "ref_log_prob": torch.full((2, 5), -2.5, device=device),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=8, global_batch_size=2)
    data = left_right_2_no_padding(data)
    labels = torch.roll(data["input_ids"].values(), -1)
    n, v, k = labels.numel(), 29, 5
    complete = torch.arange(n, device=device) % 2 == 0
    ids = (labels.unsqueeze(-1) + torch.arange(k, device=device)) % v
    fallback_ids = (labels.unsqueeze(-1) + 1 + torch.arange(k, device=device)) % v
    ids = torch.where(complete.unsqueeze(-1), ids, fallback_ids)
    generator = torch.Generator(device=device).manual_seed(8196)
    q_full = torch.log_softmax(torch.randn(n, v, generator=generator, device=device), -1)
    q_head = q_full.gather(-1, ids)
    q_support = q_head.masked_fill(torch.arange(k, device=device).unsqueeze(0) >= 3, float("-inf"))
    q_support -= q_support.logsumexp(-1, keepdim=True)
    q_head = torch.where(complete.unsqueeze(-1), q_support, q_head)
    q_sample = torch.where(complete, q_support[:, 0], q_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1))
    offsets = data["input_ids"].offsets()
    data["rollout_topk_ids"] = torch.nested.nested_tensor_from_jagged(ids.to(torch.int32), offsets)
    data["rollout_topk_log_probs"] = torch.nested.nested_tensor_from_jagged(q_head, offsets)
    data["rollout_support_complete"] = torch.nested.nested_tensor_from_jagged(complete, offsets)
    # Bypass treats these as actual sampler probabilities for response prediction rows.
    data["old_log_probs"] = torch.nested.as_nested_tensor([q_sample[2:7], q_sample[9:12]], layout=torch.jagged)
    return data, labels, ids, q_head, complete


def dense_model_inputs(data):
    rows = data["input_ids"].unbind()
    maximum = max(row.numel() for row in rows)
    input_ids = torch.stack([torch.nn.functional.pad(row, (0, maximum - row.numel())) for row in rows])
    attention_mask = torch.stack(
        [torch.nn.functional.pad(torch.ones_like(row), (0, maximum - row.numel())) for row in rows]
    )
    return input_ids, attention_mask


def packed_model_logits(logits, data):
    lengths = data["input_ids"].offsets().diff().tolist()
    return torch.cat([logits[i, :length] for i, length in enumerate(lengths)], dim=0).unsqueeze(0)


def make_actor_config(auxiliary=False):
    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(
            loss_mode="bypass_mode",
            rollout_correction=RolloutCorrectionConfig(
                bypass_mode=True, loss_type="reinforce", rollout_is="token", score_centering=True
            ),
        ),
        loss_agg_mode="token-mean",
        entropy_coeff=0.02 if auxiliary else 0.0,
        use_kl_loss=auxiliary,
    )
    config.global_batch_info.update(dp_size=1, batch_num_tokens=None, global_batch_size=None, loss_scale_factor=None)
    return config


def raw_reference_scores(logits, labels, ids, q_head, complete):
    rule = lambda ratio: ratio.clamp(max=2.0)
    lp = torch.log_softmax(logits.float(), -1)
    head = lp.gather(-1, ids)
    proxy, sampler_mass, proxy_mass = score_centering_correction(head, q_head, rule)
    support, log_mass, _ = score_centering_support_correction(head, q_head, rule)
    sampled = lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return {
        "log_probs": sampled - torch.where(complete, log_mass, torch.zeros_like(log_mass)),
        "sc_correction": torch.where(complete, support, proxy),
        "sc_sampler_head_mass": sampler_mass,
        "sc_train_head_mass": torch.where(complete, log_mass.detach().exp(), proxy_mass),
    }


@pytest.mark.parametrize("use_remove_padding", [False, True])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_shared_fsdp_actor_loss_preserves_tiny_model_parameter_gradients(
    use_remove_padding, auxiliary, device="cpu", dtype=torch.float32
):
    from verl.trainer.ppo.score_centering import score_centering_shared_scores_from_logits

    torch.manual_seed(8196)
    model_config = Qwen2Config(
        vocab_size=29,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model_config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(model_config).to(device=device, dtype=dtype).train()
    data, labels, ids, q_head, complete = actor_fixture(device)
    tu.assign_non_tensor(
        data,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        score_centering=True,
        calculate_entropy=auxiliary,
    )
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    engine.engine_config = SimpleNamespace(
        entropy_checkpointing=False,
        entropy_from_logits_with_chunking=False,
    )
    engine.compute_entropy_from_logits = verl_F.entropy_from_logits
    output_args = {
        "input_ids_rmpad_rolled": labels,
        "temperature_rmpad": torch.ones(labels.numel(), device=device),
        "temperature": torch.ones(2, device=device),
        "temperature_is_one": True,
        "pad_size": 0,
    }

    def run(candidate):
        model.zero_grad(set_to_none=True)
        model_ids, model_attention = dense_model_inputs(data)
        output = model(input_ids=model_ids, attention_mask=model_attention, use_cache=False)
        if use_remove_padding:
            output.logits = packed_model_logits(output.logits, data)
        original_logits = output.logits.detach().clone()

        def hook(student_logits, data, sampled_token_ids=None, **kwargs):
            assert sampled_token_ids is not None, "FSDP must pass its globally rolled, correctly sliced labels."
            logits = student_logits.squeeze(0)
            if candidate:
                result = score_centering_shared_scores_from_logits(
                    logits,
                    ids,
                    sampled_token_ids,
                    q_head,
                    complete,
                    lambda ratio: ratio.clamp(max=2.0),
                    chunk_size=3,
                )
            else:
                result = raw_reference_scores(logits, sampled_token_ids, ids, q_head, complete)
            return {key: value.unsqueeze(0) for key, value in result.items()}

        with patch(
            "verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits",
            side_effect=AssertionError("Shared hook already provided log_probs; CE must be skipped."),
        ):
            prepared = engine.prepare_model_outputs(output, output_args, data, hook)
        actor_loss, metrics = ppo_loss(make_actor_config(auxiliary), prepared, data)
        actor_loss.backward()
        assert torch.equal(output.logits.detach(), original_logits)
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        return actor_loss.detach(), metrics, gradients

    baseline_loss, baseline_metrics, baseline_gradients = run(False)
    actual_loss, actual_metrics, actual_gradients = run(True)
    torch.testing.assert_close(actual_loss, baseline_loss, rtol=2e-5, atol=3e-6)
    assert actual_gradients.keys() == baseline_gradients.keys()
    for name, grad in actual_gradients.items():
        torch.testing.assert_close(
            grad,
            baseline_gradients[name],
            rtol=0.03 if dtype == torch.bfloat16 else 3e-5,
            atol=2e-3 if dtype == torch.bfloat16 else 3e-6,
        )
        assert torch.isfinite(grad).all()
    for name in ("actor/sc_sampler_head_mass", "actor/sc_train_head_mass"):
        assert actual_metrics[name].aggregate() == pytest.approx(baseline_metrics[name].aggregate(), abs=3e-6)
