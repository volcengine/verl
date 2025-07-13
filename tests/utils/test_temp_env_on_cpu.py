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

import os
import unittest

from verl.utils.py_functional import temp_env_var


class TestTempEnvVarOnCPU(unittest.TestCase):
    """Test cases for temp_env_var context manager on CPU.

    Test Plan:
    1. Test setting and restoring environment variables that didn't exist before
    2. Test setting and restoring environment variables that already existed
    3. Test that environment variables are restored even when exceptions occur
    4. Test multiple nested context managers
    5. Test that original environment state is preserved after context exit
    """

    def setUp(self):
        # Store original environment state to restore after each test
        self.original_env = dict(os.environ)

        # Clean up any test variables that might exist
        test_vars = ["TEST_VAR", "TEST_VAR_2", "EXISTING_VAR"]
        for var in test_vars:
            if var in os.environ:
                del os.environ[var]

    def tearDown(self):
        # Restore original environment state
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_set_new_env_var(self):
        """Test setting a new environment variable that didn't exist before."""
        # Ensure variable doesn't exist
        self.assertNotIn("TEST_VAR", os.environ)

        with temp_env_var("TEST_VAR", "test_value"):
            # Variable should be set inside context
            self.assertEqual(os.environ["TEST_VAR"], "test_value")
            self.assertIn("TEST_VAR", os.environ)

        # Variable should be removed after context
        self.assertNotIn("TEST_VAR", os.environ)

    def test_restore_existing_env_var(self):
        """Test restoring an environment variable that already existed."""
        # Set up existing variable
        os.environ["EXISTING_VAR"] = "original_value"

        with temp_env_var("EXISTING_VAR", "temporary_value"):
            # Variable should be temporarily changed
            self.assertEqual(os.environ["EXISTING_VAR"], "temporary_value")

        # Variable should be restored to original value
        self.assertEqual(os.environ["EXISTING_VAR"], "original_value")

    def test_env_var_restored_on_exception(self):
        """Test that environment variables are restored even when exceptions occur."""
        # Set up existing variable
        os.environ["EXISTING_VAR"] = "original_value"

        with self.assertRaises(ValueError):
            with temp_env_var("EXISTING_VAR", "temporary_value"):
                # Verify variable is set
                self.assertEqual(os.environ["EXISTING_VAR"], "temporary_value")
                # Raise exception
                raise ValueError("Test exception")

        # Variable should still be restored despite exception
        self.assertEqual(os.environ["EXISTING_VAR"], "original_value")

    def test_nested_context_managers(self):
        """Test nested temp_env_var context managers."""
        # Set up original variable
        os.environ["TEST_VAR"] = "original"

        with temp_env_var("TEST_VAR", "level1"):
            self.assertEqual(os.environ["TEST_VAR"], "level1")

            with temp_env_var("TEST_VAR", "level2"):
                self.assertEqual(os.environ["TEST_VAR"], "level2")

            # Should restore to level1
            self.assertEqual(os.environ["TEST_VAR"], "level1")

        # Should restore to original
        self.assertEqual(os.environ["TEST_VAR"], "original")

    def test_multiple_different_vars(self):
        """Test setting multiple different environment variables."""
        # Set up one existing variable
        os.environ["EXISTING_VAR"] = "existing_value"

        with temp_env_var("EXISTING_VAR", "modified"):
            with temp_env_var("TEST_VAR", "new_value"):
                self.assertEqual(os.environ["EXISTING_VAR"], "modified")
                self.assertEqual(os.environ["TEST_VAR"], "new_value")

        # Check restoration
        self.assertEqual(os.environ["EXISTING_VAR"], "existing_value")
        self.assertNotIn("TEST_VAR", os.environ)

    def test_empty_string_value(self):
        """Test setting environment variable to empty string."""
        with temp_env_var("TEST_VAR", ""):
            self.assertEqual(os.environ["TEST_VAR"], "")
            self.assertIn("TEST_VAR", os.environ)

        # Should be removed after context
        self.assertNotIn("TEST_VAR", os.environ)

    def test_overwrite_with_empty_string(self):
        """Test overwriting existing variable with empty string."""
        os.environ["EXISTING_VAR"] = "original"

        with temp_env_var("EXISTING_VAR", ""):
            self.assertEqual(os.environ["EXISTING_VAR"], "")

        # Should restore original value
        self.assertEqual(os.environ["EXISTING_VAR"], "original")

    def test_context_manager_returns_none(self):
        """Test that context manager yields None."""
        with temp_env_var("TEST_VAR", "value") as result:
            self.assertIsNone(result)
            self.assertEqual(os.environ["TEST_VAR"], "value")


if __name__ == "__main__":
    unittest.main()
