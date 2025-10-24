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

import unittest
from unittest.mock import patch

from verl.utils.tracking import _MlflowLoggingAdapter


class TestMlflowLoggingAdapter(unittest.TestCase):
    def test_sanitize_key_and_warning(self):
        adapter = _MlflowLoggingAdapter()
        data = {"valid_key": 1.0, "invalid@key!": 2.0, "another/valid-key": 3.0, "bad key#": 4.0}
        # Patch mlflow.log_metrics to capture the metrics actually sent
        with (
            patch("mlflow.log_metrics") as mock_log_metrics,
            patch.object(adapter, "logger") as mock_logger,
        ):
            adapter.log(data, step=5)
            # Check that keys are sanitized
            sent_metrics = mock_log_metrics.call_args[1]["metrics"]
            self.assertIn("invalid_at_key_", sent_metrics)  # @ becomes _at_, ! becomes _
            self.assertIn("bad key_", sent_metrics)  # # becomes _, space remains
            self.assertNotIn("invalid@key!", sent_metrics)
            self.assertNotIn("bad key#", sent_metrics)
            # Check that a warning was logged for each sanitized key
            warning_msgs = [str(call) for call in mock_logger.warning.call_args_list]
            self.assertTrue(any("invalid@key!" in msg and "invalid_at_key_" in msg for msg in warning_msgs))
            self.assertTrue(any("bad key#" in msg and "bad key_" in msg for msg in warning_msgs))


if __name__ == "__main__":
    unittest.main()
