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

import torch
import torch.nn as nn


class SVDLinear(nn.Module):
    # Implementation of SVD-LoRA-GRPO method introduced in [ESSA: Evolutionary Strategies for Scalable Alignment](https://arxiv.org/abs/2507.04453)
    U: torch.Tensor
    sigma: torch.Tensor
    V: torch.Tensor

    def __init__(
        self,
        U: torch.Tensor,
        sigma: torch.Tensor,
        V: torch.Tensor,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        super().__init__()

        if device is not None:
            U = U.to(device)
            sigma = sigma.to(device)
            V = V.to(device)

        U = U.contiguous()
        sigma = sigma.contiguous()
        V = V.contiguous()

        self.U = nn.Parameter(U, requires_grad=False).to(torch.float32)
        self.sigma = nn.Parameter(sigma, requires_grad=True).to(torch.float32)
        self.V = nn.Parameter(V, requires_grad=False).to(torch.float32)

        self.out_features = self.U.size(0)
        self.in_features = self.V.size(0)
        self.bias = None

    @staticmethod
    def create_from_weight(weight: torch.Tensor) -> "SVDLinear":
        U, S, Vh = torch.linalg.svd(weight.to(torch.float32), full_matrices=True)
        V = Vh.T
        return SVDLinear(U, S, V, dtype=weight.dtype, device=weight.device)

    def _get_svd_weight(self) -> torch.Tensor:
        W = (self.U * self.sigma) @ self.V.T
        return W.contiguous()

    @property
    def weight(self) -> torch.Tensor:
        return self._get_svd_weight()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input @ self.weight.T


def apply_svd_lora(model: nn.Module) -> nn.Module:
    def set_deep_attr(obj, attr_path, value):
        parts = attr_path.split(".")
        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        setattr(obj, parts[-1], value)

    replacements = {}

    for name, mod in model.named_modules():
        if ("lora_A" in name or "lora_B" in name) and not name.endswith(".default"):
            if isinstance(mod, nn.ModuleDict):
                new_dict = nn.ModuleDict()
                for key, sub in mod.items():
                    W = sub.weight.data
                    new_dict[key] = SVDLinear.create_from_weight(W)
                replacements[name] = new_dict

    for name, new_mod in replacements.items():
        set_deep_attr(model, name, new_mod)

    return model
