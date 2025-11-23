#!/usr/bin/env python3
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
"""
Unit tests for get_tool_call_parser_type function.

Tests that get_tool_call_parser_type() correctly identifies models using model_type
and returns the correct parser, avoiding false matches from vocabulary overlap.

This addresses Issue #4203 where Qwen2.5 models were incorrectly detected as GLM4
due to shared special tokens in their vocabularies.

Related Issue: https://github.com/volcengine/verl/issues/4203
"""

import unittest
from unittest.mock import MagicMock, Mock


class MockTokenizer:
    """Mock tokenizer for testing"""

    def __init__(self, name_or_path, vocab=None, model_type=None):
        self.name_or_path = name_or_path
        self._vocab = vocab or {}
        self.model_type = model_type

    def get_vocab(self):
        return self._vocab


class TestGetToolCallParserType(unittest.TestCase):
    """Test cases for get_tool_call_parser_type function"""

    def setUp(self):
        """Set up test fixtures"""
        # Common vocabulary that might exist in multiple models
        self.common_vocab = {
            "<|assistant|>": 1,
            "<|user|>": 2,
            "<|endoftext|>": 3,
            "<|observation|>": 4,
        }

    def test_model_type_detection_qwen2(self):
        """Test that model_type='qwen2' is detected as qwen25 (highest priority)"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        tokenizer = MockTokenizer(
            name_or_path="Qwen/Qwen2.5-3B-Instruct",
            vocab=self.common_vocab,
            model_type="qwen2"  # Qwen2.5 uses model_type="qwen2"
        )

        parser_type = get_tool_call_parser_type(tokenizer)
        self.assertEqual(
            parser_type,
            "qwen25",
            "model_type='qwen2' should be detected as 'qwen25'"
        )

    def test_model_type_detection_qwen2_vl(self):
        """Test that model_type='qwen2_vl' is detected as qwen25"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        tokenizer = MockTokenizer(
            name_or_path="Qwen/Qwen2-VL-7B-Instruct",
            vocab=self.common_vocab,
            model_type="qwen2_vl"
        )

        parser_type = get_tool_call_parser_type(tokenizer)
        self.assertEqual(parser_type, "qwen25")

    def test_model_type_detection_glm4(self):
        """Test that model_type='glm4' is detected correctly"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        tokenizer = MockTokenizer(
            name_or_path="THUDM/glm-4-9b-chat",
            vocab=self.common_vocab,
            model_type="glm4"
        )

        parser_type = get_tool_call_parser_type(tokenizer)
        self.assertEqual(parser_type, "glm4")

    def test_model_type_takes_precedence_over_name(self):
        """Test that model_type detection has higher priority than name matching"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        # Even with a confusing name, model_type should win
        tokenizer = MockTokenizer(
            name_or_path="custom/my-glm4-style-qwen2.5-model",  # Confusing name
            vocab=self.common_vocab,
            model_type="qwen2"  # But model_type is qwen2
        )

        parser_type = get_tool_call_parser_type(tokenizer)
        self.assertEqual(
            parser_type,
            "qwen25",
            "model_type should take precedence over name pattern matching"
        )

    def test_qwen25_3b_detection(self):
        """Test that Qwen2.5-3B is detected as qwen25 via name (fallback)"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        tokenizer = MockTokenizer(
            name_or_path="Qwen/Qwen2.5-3B-Instruct",
            vocab=self.common_vocab
        )

        parser_type = get_tool_call_parser_type(tokenizer)
        self.assertEqual(
            parser_type,
            "qwen25",
            f"Qwen2.5-3B should be detected as 'qwen25', got '{parser_type}'"
        )

    def test_qwen25_7b_detection(self):
        """Test that Qwen2.5-7B is detected as qwen25"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        tokenizer = MockTokenizer(
            name_or_path="Qwen/Qwen2.5-7B-Instruct",
            vocab=self.common_vocab
        )

        parser_type = get_tool_call_parser_type(tokenizer)
        self.assertEqual(parser_type, "qwen25")

    def test_qwen25_with_dash(self):
        """Test that qwen-2.5 (with dash) is also detected"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        tokenizer = MockTokenizer(
            name_or_path="qwen-2.5-14b-instruct",
            vocab=self.common_vocab
        )

        parser_type = get_tool_call_parser_type(tokenizer)
        self.assertEqual(parser_type, "qwen25")

    def test_qwen25_case_insensitive(self):
        """Test that detection is case-insensitive"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        # Test uppercase
        tokenizer = MockTokenizer(
            name_or_path="QWEN/QWEN2.5-3B-INSTRUCT",
            vocab=self.common_vocab
        )

        parser_type = get_tool_call_parser_type(tokenizer)
        self.assertEqual(parser_type, "qwen25")

        # Test mixed case
        tokenizer = MockTokenizer(
            name_or_path="Qwen/QwEn2.5-7B-InStRuCt",
            vocab=self.common_vocab
        )

        parser_type = get_tool_call_parser_type(tokenizer)
        self.assertEqual(parser_type, "qwen25")

    def test_glm4_not_matched_as_qwen25(self):
        """Test that GLM-4 models are NOT matched as qwen25"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        tokenizer = MockTokenizer(
            name_or_path="THUDM/glm-4-9b-chat",
            vocab=self.common_vocab
        )

        parser_type = get_tool_call_parser_type(tokenizer)
        # Should not be qwen25
        self.assertNotEqual(
            parser_type,
            "qwen25",
            "GLM-4 should not be detected as qwen25"
        )

    def test_gpt_oss_takes_precedence(self):
        """Test that gpt-oss check still works (regression test)"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        tokenizer = MockTokenizer(
            name_or_path="gpt-oss-instruct",
            vocab=self.common_vocab
        )

        parser_type = get_tool_call_parser_type(tokenizer)
        self.assertEqual(
            parser_type,
            "gpt-oss",
            "gpt-oss detection should still work"
        )

    def test_qwen25_path_variations(self):
        """Test various Qwen2.5 path formats"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        test_paths = [
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-14B",
            "Qwen/Qwen2.5-32B",
            "Qwen/Qwen2.5-72B",
            "qwen2.5-coder-7b",
            "custom/qwen-2.5-finetuned",
            "/path/to/qwen2.5-local",
        ]

        for path in test_paths:
            with self.subTest(path=path):
                tokenizer = MockTokenizer(
                    name_or_path=path,
                    vocab=self.common_vocab
                )

                parser_type = get_tool_call_parser_type(tokenizer)
                self.assertEqual(
                    parser_type,
                    "qwen25",
                    f"Path '{path}' should be detected as qwen25"
                )

    def test_processor_support(self):
        """Test that both tokenizer and processor work"""
        from verl.workers.rollout.sglang_rollout.sglang_rollout import get_tool_call_parser_type

        # Mock processor (like Qwen2-VL)
        class MockProcessor:
            def __init__(self, name_or_path, vocab, model_type=None):
                self.name_or_path = name_or_path
                self.tokenizer = MockTokenizer(name_or_path, vocab, model_type)

        processor = MockProcessor(
            name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            vocab=self.common_vocab,
            model_type="qwen2_vl"
        )

        parser_type = get_tool_call_parser_type(processor)
        self.assertEqual(parser_type, "qwen25")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGetToolCallParserType))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())
