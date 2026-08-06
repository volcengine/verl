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

"""Round-trip tests for the mcore fused-QKV un-fusing helper.

These run on CPU: ``unfuse_mcore_qkv`` is a pure tensor function and must not
require a GPU/NPU, megatron, or a real checkpoint.

The forward direction lives in ``verl/models/mcore/loader.py`` (see
``_broadcast_tp_shard_tensor_qkv``): for TP=1 the fused tensor is interleaved
per query group as ``[q_heads_of_group, k_head, v_head]`` rather than being a
plain ``cat([q, k, v])``. A verifier that assumed the plain layout would report
false mismatches on every GQA model, so the inverse is pinned down here.
"""

import unittest

import torch

from scripts.verify_hf_to_mcore import unfuse_mcore_qkv


def fuse_reference(q, k, v, num_attention_heads, num_key_value_heads, head_dim):
    """Mirror of verl/models/mcore/loader.py:311-324 for tp_size == 1."""
    q_per_group = (num_attention_heads // num_key_value_heads) * head_dim
    blocks = []
    for j in range(num_key_value_heads):
        blocks.append(q[j * q_per_group : (j + 1) * q_per_group])
        blocks.append(k[j * head_dim : (j + 1) * head_dim])
        blocks.append(v[j * head_dim : (j + 1) * head_dim])
    return torch.cat(blocks, dim=0)


class TestUnfuseMcoreQkv(unittest.TestCase):
    # Qwen2.5-Math-7B geometry: 28 heads over 4 KV groups, head_dim 128.
    HEADS = 28
    KV_HEADS = 4
    HEAD_DIM = 128
    HIDDEN = 3584

    def _round_trip(self, weight_2d: bool):
        torch.manual_seed(0)
        q_rows = self.HEADS * self.HEAD_DIM
        kv_rows = self.KV_HEADS * self.HEAD_DIM
        shape = (lambda n: (n, self.HIDDEN)) if weight_2d else (lambda n: (n,))
        q = torch.randn(*shape(q_rows))
        k = torch.randn(*shape(kv_rows))
        v = torch.randn(*shape(kv_rows))

        fused = fuse_reference(q, k, v, self.HEADS, self.KV_HEADS, self.HEAD_DIM)
        got_q, got_k, got_v = unfuse_mcore_qkv(fused, self.HEADS, self.KV_HEADS, self.HEAD_DIM)

        for name, want, got in (("q", q, got_q), ("k", k, got_k), ("v", v, got_v)):
            self.assertEqual(got.shape, want.shape, f"{name} shape")
            self.assertTrue(torch.equal(got, want), f"{name} content")

    def test_round_trip_weight(self):
        self._round_trip(weight_2d=True)

    def test_round_trip_bias(self):
        self._round_trip(weight_2d=False)

    def test_multi_head_attention_is_a_special_case(self):
        """kv_heads == heads (no GQA) must also round-trip."""
        torch.manual_seed(1)
        heads = kv_heads = 4
        head_dim = 8
        rows = heads * head_dim
        q, k, v = (torch.randn(rows, 16) for _ in range(3))
        fused = fuse_reference(q, k, v, heads, kv_heads, head_dim)
        got_q, got_k, got_v = unfuse_mcore_qkv(fused, heads, kv_heads, head_dim)
        self.assertTrue(torch.equal(got_q, q))
        self.assertTrue(torch.equal(got_k, k))
        self.assertTrue(torch.equal(got_v, v))

    def test_plain_split_would_be_wrong(self):
        """Guard the regression this helper exists for.

        A naive cat([q, k, v]) split must NOT reproduce the interleaved layout,
        otherwise the test above would pass for the wrong reason.
        """
        torch.manual_seed(2)
        q = torch.randn(self.HEADS * self.HEAD_DIM, 8)
        k = torch.randn(self.KV_HEADS * self.HEAD_DIM, 8)
        v = torch.randn(self.KV_HEADS * self.HEAD_DIM, 8)
        fused = fuse_reference(q, k, v, self.HEADS, self.KV_HEADS, self.HEAD_DIM)
        naive_q = fused[: q.shape[0]]
        self.assertFalse(torch.equal(naive_q, q))

    def test_rejects_inconsistent_geometry(self):
        with self.assertRaises(ValueError):
            unfuse_mcore_qkv(torch.zeros(10, 4), self.HEADS, self.KV_HEADS, self.HEAD_DIM)


if __name__ == "__main__":
    unittest.main()
