# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.workers.rollout.r3_utils import (
    decode_kimi_full_r3_payload,
    get_kimi_full_r3_topk,
    is_kimi_full_r3_config,
)


def _pack_exact_bf16(ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Exercise the same BF16 byte split used by the Ascend capturer."""
    ids_tensor = torch.from_numpy(ids.astype(np.int32))
    weights_tensor = torch.from_numpy(weights).to(torch.bfloat16)
    weight_bytes = weights_tensor.contiguous().view(torch.uint8).reshape(*weights_tensor.shape, 2)
    packed = torch.cat(
        (
            ids_tensor,
            weight_bytes[..., 0].to(torch.int32),
            weight_bytes[..., 1].to(torch.int32),
        ),
        dim=-1,
    )
    return packed.numpy()


def test_full_r3_payload_round_trips_ids_and_executed_bf16_weights():
    ids = np.array(
        [
            [[[1, 7], [2, 6]], [[3, 5], [0, 4]]],
            [[[7, 1], [6, 2]], [[5, 3], [4, 0]]],
        ],
        dtype=np.uint16,
    ).reshape(4, 2, 2)
    # These values are exactly representable in BF16, so the expected result
    # is unambiguous and bit-preserving through FP32 decode.
    weights = np.array(
        [0.25, 0.5, 1.0, 2.0] * 4,
        dtype=np.float32,
    ).reshape(ids.shape)

    decoded_ids, decoded_weights = decode_kimi_full_r3_payload(
        _pack_exact_bf16(ids, weights),
        expected_topk=2,
    )

    np.testing.assert_array_equal(decoded_ids, ids)
    np.testing.assert_array_equal(decoded_weights, weights)


def test_896_expert_routes_survive_scheduler_uint16_and_megatron_transport():
    from verl.workers.rollout.r3_utils import pack_kimi_full_r3_for_megatron

    ids = np.array([[[0, 255, 256, 895]]], dtype=np.uint16)
    weights = np.array([[[0.125, 0.25, 0.5, 1.0]]], dtype=np.float32)
    payload = _pack_exact_bf16(ids, weights).astype(np.uint16)
    decoded_ids, decoded_weights = decode_kimi_full_r3_payload(payload, expected_topk=4)
    assert decoded_ids.dtype == np.int32
    routes = torch.nested.as_nested_tensor([torch.from_numpy(decoded_ids)], layout=torch.jagged)
    route_weights = torch.nested.as_nested_tensor([torch.from_numpy(decoded_weights)], layout=torch.jagged)
    packed = pack_kimi_full_r3_for_megatron(routes, route_weights)
    assert packed.dtype == torch.int32
    actual_ids, actual_weights = decode_kimi_full_r3_payload(packed.values().numpy(), expected_topk=4)
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_array_equal(actual_weights, weights)


@pytest.mark.parametrize("expert_ids", [[0, 7], [256, 895]])
def test_kimi_dense_padding_preserves_int32_routes(expert_ids):
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.workers.rollout.r3_utils import KIMI_FULL_MODEL_ROUTE_SEMANTICS
    from verl.workers.utils.padding import left_right_2_no_padding

    ids = torch.tensor(expert_ids, dtype=torch.int32).reshape(1, 1, 1, 2).expand(2, 5, 1, 2)
    data = TensorDict(
        dict(
            input_ids=torch.ones(2, 4, dtype=torch.long),
            attention_mask=torch.ones(2, 4),
            response_mask=torch.ones(2, 2),
            position_ids=torch.arange(4).expand(2, 4),
            routed_experts=ids,
            routed_expert_weights=torch.full(ids.shape, 0.5),
        ),
        batch_size=[2],
    )
    tu.assign_non_tensor_data(data, "routed_experts_transport_semantics", KIMI_FULL_MODEL_ROUTE_SEMANTICS)
    result = left_right_2_no_padding(data)
    torch.testing.assert_close(result["routed_experts"].values(), ids.flatten(0, 1))
    assert result["routed_experts"].dtype == torch.int32
    assert result["routed_experts"].offsets().tolist() == [0, 5, 10]


def test_id_only_payload_is_rejected_instead_of_silently_downgrading_r3():
    ids = np.zeros((3, 4, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="weight capture is not active"):
        decode_kimi_full_r3_payload(ids, expected_topk=2)


def test_corrupt_weight_byte_lane_is_rejected():
    payload = np.zeros((1, 1, 6), dtype=np.int32)
    payload[..., 2] = 256
    with pytest.raises(ValueError, match="byte lanes are out of range"):
        decode_kimi_full_r3_payload(payload, expected_topk=2)


def test_topk_is_resolved_from_nested_kimi_text_config():
    config = SimpleNamespace(
        model_type="kimi_k3",
        text_config=SimpleNamespace(
            model_type="kimi_linear",
            num_experts_per_token=8,
        ),
    )
    assert get_kimi_full_r3_topk(config) == 8
    assert is_kimi_full_r3_config(config)


def test_non_kimi_config_keeps_legacy_id_only_schema():
    config = SimpleNamespace(
        model_type="qwen3_moe",
        num_experts_per_tok=8,
    )
    assert not is_kimi_full_r3_config(config)
    with pytest.raises(ValueError, match="only for Kimi K3"):
        get_kimi_full_r3_topk(config)


def test_megatron_boundary_preserves_jagged_rows_and_bf16_bits():
    from verl.workers.rollout.r3_utils import pack_kimi_full_r3_for_megatron

    ids = torch.nested.as_nested_tensor(
        [torch.tensor([[[1, 7]], [[3, 5]], [[4, 2]]]), torch.tensor([[[6, 0]]])],
        layout=torch.jagged,
    )
    weights = torch.nested.nested_tensor_from_jagged(
        torch.tensor([[[0.25, 0.5]], [[1.0, 2.0]], [[0.125, 4.0]], [[0.5, 0.5]]]),
        offsets=ids.offsets(),
    )
    packed = pack_kimi_full_r3_for_megatron(ids, weights)
    assert torch.equal(packed.offsets(), ids.offsets())
    decoded_ids, decoded_weights = decode_kimi_full_r3_payload(packed.values().numpy(), expected_topk=2)
    np.testing.assert_array_equal(decoded_ids, ids.values().numpy())
    np.testing.assert_array_equal(decoded_weights, weights.values().numpy())
    with pytest.raises(ValueError, match="both expert IDs and weights"):
        pack_kimi_full_r3_for_megatron(ids, None)
    with pytest.raises(ValueError, match="exact BF16"):
        pack_kimi_full_r3_for_megatron(ids, weights + 0.00001)


@pytest.mark.parametrize("invalid", [None, "duplicate", "range", "nan", "negative", "zero"])
def test_full_route_validation_before_agent_transport(invalid):
    from verl.workers.rollout.r3_utils import validate_kimi_full_model_routes

    config = SimpleNamespace(model_type="kimi_k3", num_hidden_layers=2, num_experts=4, num_experts_per_token=2)
    ids = torch.tensor([[[0, 1], [2, 3]]])
    weights = torch.full(ids.shape, 0.5)
    if invalid == "duplicate":
        ids[0, 1, 0] = 3
    elif invalid == "range":
        ids[0, 1, 0] = 4
    elif invalid == "nan":
        weights[0, 1, 0] = float("nan")
    elif invalid == "negative":
        weights[0, 1, 0] = -0.5
    elif invalid == "zero":
        weights[0, 1] = 0
    if invalid is None:
        validate_kimi_full_model_routes(ids, weights, config)
    else:
        with pytest.raises(ValueError):
            validate_kimi_full_model_routes(ids, weights, config)
