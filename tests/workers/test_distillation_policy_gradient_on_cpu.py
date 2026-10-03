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

"""Compare real policy-gradient distillation against a dense PPO reference."""

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import distillation_loss
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.config.distillation import DistillationConfig, DistillationLossConfig


def _nested(rows: list[list[float]] | list[list[int]] | list[list[bool]]) -> torch.Tensor:
    return torch.nested.nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)


@pytest.mark.parametrize("weights_layout", ["nested", "dense", "absent"])
@pytest.mark.parametrize("nested_response_fields", [True, False])
@pytest.mark.parametrize("loss_agg_mode", ["token-mean", "seq-mean-token-mean"])
def test_policy_gradient_weights_match_dense_loss_and_gradient(
    weights_layout: str, nested_response_fields: bool, loss_agg_mode: str
) -> None:
    """Unequal response lengths, masked tokens and non-unit weights preserve PPO semantics."""
    actor_config = ActorConfig(
        strategy="megatron", rollout_n=1, ppo_micro_batch_size_per_gpu=1, loss_agg_mode=loss_agg_mode
    )
    loss_config = DistillationLossConfig(loss_mode="k1", loss_max_clamp=None, use_policy_gradient=True)
    distillation_config = DistillationConfig(distillation_loss=loss_config)
    # Sequence lengths are 5 and 3, with prompt lengths 2 and 1. Model outputs
    # predict the next token; response log-probs are at packed indices 1:4 and 5:7.
    student = torch.tensor([-9.0, -0.2, -1.3, -0.4, -8.0, -0.6, -0.9, -7.0], requires_grad=True)
    teacher = torch.tensor([-6.0, -0.4, -1.0, -0.2, -5.0, -0.8, -0.6, -4.0], requires_grad=True)
    offsets = torch.tensor([0, 5, 8])
    mask = _nested([[True, False, True], [True, True]])
    old_log_probs = _nested([[-0.25, -1.1, -0.5], [-0.65, -0.7]])
    dense_mask = torch.tensor([[True, False, True], [True, True, False]])
    dense_old_log_probs = torch.tensor([[-0.25, -1.1, -0.5], [-0.65, -0.7, 0.0]])
    data = TensorDict(
        {
            "prompts": _nested([[1, 2], [3]]),
            "responses": _nested([[4, 5, 6], [7, 8]]),
            "response_mask": mask if nested_response_fields else dense_mask,
            "old_log_probs": old_log_probs if nested_response_fields else dense_old_log_probs,
            "teacher_logprobs": torch.nested.nested_tensor_from_jagged(teacher.unsqueeze(-1), offsets),
        },
        batch_size=[],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=4, global_batch_size=2)
    dense_weights = torch.tensor([[1.3, 0.7, 0.0], [0.6, 1.8, 0.0]])
    if weights_layout == "nested":
        data["rollout_is_weights"] = _nested([[1.3, 0.7, 0.0], [0.6, 1.8]])
    elif weights_layout == "dense":
        data["rollout_is_weights"] = dense_weights
    model_output = {"log_probs": torch.nested.nested_tensor_from_jagged(student, offsets)}
    actual_loss, metrics = distillation_loss(actor_config, distillation_config, model_output, data)
    actual_loss.backward()

    # Build the reference independently of no_padding_2_padding and the weights
    # conversion, while exercising the same real clipped policy-loss function.
    reference_log_probs = torch.tensor([[-0.2, -1.3, -0.4], [-0.6, -0.9, 0.0]], requires_grad=True)
    teacher_log_probs = torch.tensor([[-0.4, -1.0, -0.2], [-0.8, -0.6, 0.0]])
    advantages = -(reference_log_probs.detach() - teacher_log_probs)
    expected_loss, expected_metrics = compute_policy_loss_vanilla(
        old_log_prob=dense_old_log_probs,
        log_prob=reference_log_probs,
        advantages=advantages,
        response_mask=dense_mask,
        loss_agg_mode=loss_agg_mode,
        config=loss_config,
        rollout_is_weights=dense_weights if weights_layout != "absent" else None,
    )
    expected_loss.backward()
    torch.testing.assert_close(actual_loss, expected_loss)
    expected_gradient = torch.zeros_like(student)
    expected_gradient[1:4] = reference_log_probs.grad[0]
    expected_gradient[5:7] = reference_log_probs.grad[1, :2]
    torch.testing.assert_close(student.grad, expected_gradient)
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None
    for key, value in expected_metrics.items():
        assert metrics[key.replace("actor/", "distillation/")] == pytest.approx(value)
    if weights_layout != "absent":
        unweighted_loss, _ = compute_policy_loss_vanilla(
            old_log_prob=dense_old_log_probs,
            log_prob=reference_log_probs,
            advantages=advantages,
            response_mask=dense_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
        )
        assert not torch.isclose(actual_loss, unweighted_loss)
        assert student.grad[3] == 0  # A valid response with zero IS weight contributes no gradient.
