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

"""Version gating for the vLLM-Ascend NPU patches.

These tests run on CPU: ``resolve_npu_vllm_patch_variant`` is a pure function
over a version string and must not require an NPU or an installed vLLM.

Regression context: the gating used to be spelled as three disjoint interval
checks (``0.13-0.14``, ``>=0.19``, ``0.11-0.13``), which left ``0.14 < v <
0.19`` uncovered. On vLLM 0.18 that silently disabled the rotary-embedding
patch, the ``FusedMoE.weight_loader`` wrapper and the Ascend A2 MoE
communication fallback -- exactly the paths a MoE model on A2 depends on.
"""

import unittest

from verl.utils.vllm.npu_vllm_patch import resolve_npu_vllm_patch_variant


class TestResolveNpuVllmPatchVariant(unittest.TestCase):
    """Every supported vLLM version must map to exactly one patch variant."""

    def test_v011_range(self):
        """0.11 <= v < 0.13 uses the v011 wrappers."""
        self.assertEqual(resolve_npu_vllm_patch_variant("0.11.0"), "v011")
        self.assertEqual(resolve_npu_vllm_patch_variant("0.12.0"), "v011")
        self.assertEqual(resolve_npu_vllm_patch_variant("0.12.9"), "v011")

    def test_v013_lower_bound(self):
        """0.13 is the first version on the v013 API shape."""
        self.assertEqual(resolve_npu_vllm_patch_variant("0.13.0"), "v013")
        self.assertEqual(resolve_npu_vllm_patch_variant("0.14.0"), "v013")

    def test_v018_is_not_a_gap(self):
        """Regression: 0.14 < v < 0.19 must not fall through to no patch.

        vllm_ascend 0.18.x exposes the v013 API shape (module-level
        ``select_moe_comm_method``, ``AscendDeviceType``), while the v011
        symbols (``AscendSocVersion``, ``get_ascend_soc_version``) are gone.
        """
        self.assertEqual(resolve_npu_vllm_patch_variant("0.15.0"), "v013")
        self.assertEqual(resolve_npu_vllm_patch_variant("0.18.0"), "v013")
        self.assertEqual(resolve_npu_vllm_patch_variant("0.18.1"), "v013")

    def test_v019_and_beyond(self):
        """Versions at or above 0.19 keep using the v013 wrappers."""
        self.assertEqual(resolve_npu_vllm_patch_variant("0.19.0"), "v013")
        self.assertEqual(resolve_npu_vllm_patch_variant("0.20.0"), "v013")
        self.assertEqual(resolve_npu_vllm_patch_variant("1.0.0"), "v013")

    def test_below_supported_range(self):
        """Anything older than 0.11 has no applicable patch."""
        self.assertIsNone(resolve_npu_vllm_patch_variant("0.10.0"))
        self.assertIsNone(resolve_npu_vllm_patch_variant("0.9.1"))

    def test_dev_and_local_version_suffixes(self):
        """Dev/local suffixes must not change the resolved variant."""
        self.assertEqual(resolve_npu_vllm_patch_variant("0.18.1.dev41+g989506be7"), "v013")
        self.assertEqual(resolve_npu_vllm_patch_variant("0.18.0+empty"), "v013")

    def test_no_version_is_unclassified_within_supported_range(self):
        """No supported version may return None -- that was the original bug."""
        for major_minor in ["0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.20"]:
            with self.subTest(version=major_minor):
                self.assertIsNotNone(
                    resolve_npu_vllm_patch_variant(f"{major_minor}.0"),
                    f"vLLM {major_minor} resolved to no patch variant",
                )


if __name__ == "__main__":
    unittest.main()
