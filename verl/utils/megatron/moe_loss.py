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

from typing import Any


def validate_bshd_moe_router_loss_compatibility(transformer_config: Any) -> None:
    """Reject unmasked BSHD routing only when an MoE aux or z-loss is active."""
    routing_types = getattr(transformer_config, "moe_router_load_balancing_type", "none")
    coefficients = getattr(transformer_config, "moe_aux_loss_coeff", 0.0)
    if isinstance(routing_types, str):
        routing_types = [routing_types]
    if not isinstance(coefficients, list):
        coefficients = [coefficients] * len(routing_types)
    has_aux_loss = any(
        routing_type in ("aux_loss", "seq_aux_loss", "global_aux_loss") and coefficient > 0
        for routing_type, coefficient in zip(routing_types, coefficients, strict=True)
    )
    has_z_loss = (getattr(transformer_config, "moe_z_loss_coeff", None) or 0.0) > 0
    if has_aux_loss or has_z_loss:
        raise ValueError(
            "BSHD with calculate_per_token_loss=True does not support active MoE aux/z loss because "
            "verl does not pass a padding mask to the router. Use THD (use_remove_padding=True), "
            "disable CP, or disable the MoE aux/z loss."
        )
