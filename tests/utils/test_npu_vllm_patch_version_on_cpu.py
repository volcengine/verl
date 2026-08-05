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

import sys
import types
import unittest
from unittest.mock import patch

from verl.utils.vllm.npu_vllm_patch import installed_vllm_version, resolve_npu_vllm_patch_variant


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


class TestInstalledVllmVersion(unittest.TestCase):
    """Version lookup must survive a partially-initialised vllm module.

    Importing vllm activates its platform plugins; on Ascend that path can
    re-enter this module before ``vllm.__version__`` is assigned, so reading
    the attribute raises AttributeError and takes the whole training job down
    during container start-up.
    """

    def test_prefers_the_declared_attribute(self):
        fake = types.ModuleType("vllm")
        fake.__version__ = "0.18.0"
        with patch.dict(sys.modules, {"vllm": fake}):
            self.assertEqual(installed_vllm_version(), "0.18.0")

    def test_falls_back_to_distribution_metadata(self):
        """No __version__ yet -> read it from the installed distribution."""
        fake = types.ModuleType("vllm")  # deliberately has no __version__
        with patch.dict(sys.modules, {"vllm": fake}):
            with patch("importlib.metadata.version", return_value="0.18.0+empty"):
                self.assertEqual(installed_vllm_version(), "0.18.0+empty")

    def test_fallback_value_still_resolves_to_a_variant(self):
        """The metadata form carries a local segment; gating must still work."""
        self.assertEqual(resolve_npu_vllm_patch_variant("0.18.0+empty"), "v013")

    def test_empty_attribute_is_treated_as_missing(self):
        fake = types.ModuleType("vllm")
        fake.__version__ = ""
        with patch.dict(sys.modules, {"vllm": fake}):
            with patch("importlib.metadata.version", return_value="0.18.0+empty"):
                self.assertEqual(installed_vllm_version(), "0.18.0+empty")


if __name__ == "__main__":
    unittest.main()
