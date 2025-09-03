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
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from verl.utils.rollout_skip import DataProto, RolloutSkip

len_prompt = 50
len_response = 100


def temp_dir():
    # Create a temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


def build_generate_fn(gen_bs, n):
    len_tokenizer = 1024

    def iterate():
        while True:
            prompt = torch.randint(len_tokenizer, size=(gen_bs, len_prompt)).repeat_interleave(n, dim=0)
            generate = torch.randint(len_tokenizer, size=(gen_bs * n, len_response))
            data = DataProto.from_dict(tensors={"prompt": prompt, "response": generate})
            yield data

    mock_infer_engine = iterate()

    def fn(batch, **kwargs):
        # Simulate the inference engine returning the next batch
        return next(mock_infer_engine)

    return fn


@pytest.fixture(params=[(32, 4), (64, 4), (64, 8)])
def mock_rollout_wg(request):
    gen_bs, n = request.param
    rollout_wg = MagicMock()

    config = MagicMock()
    config.actor_rollout_ref.rollout = {
        "n": n,
        "skip_dump_dir": next(temp_dir()),
    }
    config.data = {"gen_batch_size": gen_bs}

    rollout_wg.generate_sequences = build_generate_fn(gen_bs, n)

    yield config, rollout_wg
    # Cleanup
    shutil.rmtree(next(temp_dir()))


class TestRolloutSkip:
    def test_initialization(self, capsys):
        """Test that RolloutSkip initializes correctly"""
        config = MagicMock()
        config.actor_rollout_ref.rollout = {
            "n": 16,
            "skip_dump_dir": "tmp/rollout_dump",
        }
        config.data = {"gen_batch_size": 128}
        mock_rollout_wg = MagicMock()
        skip = RolloutSkip(config, mock_rollout_wg)

        assert skip.n == 16
        assert skip.gbs == 128
        assert str(skip.dumped_dir) == "tmp/rollout_dump"

        assert skip._rollout_wg == mock_rollout_wg
        skip.wrap_generate_sequences()
        captured = capsys.readouterr()
        assert "Successfully patched" in captured.out

    def test_generate_without_wrap(self, mock_rollout_wg):
        """Test that generate_sequences works without wrapping"""

        config, rollout_wg = mock_rollout_wg
        _ = RolloutSkip(config, rollout_wg)

        _result = rollout_wg.generate_sequences(MagicMock())
        for _ in range(10):
            result = rollout_wg.generate_sequences(MagicMock())
            assert isinstance(result, DataProto)
            # * make sure the data is different
            assert torch.abs(_result.batch["prompt"] - result.batch["prompt"]).sum() > 0
            assert torch.abs(_result.batch["response"] - result.batch["response"]).sum() > 0
            _result = result

    def test_dump(self, mock_rollout_wg, capsys):
        config, rollout_wg = mock_rollout_wg
        skip = RolloutSkip(config, rollout_wg)
        skip.wrap_generate_sequences()

        result = rollout_wg.generate_sequences(MagicMock())
        # * check if dump is OK
        assert skip.curr_path_dump.exists()
        captured = capsys.readouterr()
        assert "Successfully dump data in" in captured.out
        # * get file size, estimate file size
        file_size = skip.curr_path_dump.stat().st_size
        est_file_size = (len_prompt + len_response) * skip.gbs * skip.n * result.batch["prompt"].dtype.itemsize
        assert file_size >= est_file_size, "Dumped file size is smaller than expected"

    def test_generate_with_wrap(self, mock_rollout_wg, capsys):
        """Test that generate_sequences works without wrapping"""

        config, rollout_wg = mock_rollout_wg
        skip = RolloutSkip(config, rollout_wg)
        skip.wrap_generate_sequences()

        _result = rollout_wg.generate_sequences(MagicMock())

        for _ in range(10):
            result = rollout_wg.generate_sequences(MagicMock())
            assert isinstance(result, DataProto)
            # * make sure the data is different
            assert torch.abs(_result.batch["prompt"] - result.batch["prompt"]).sum() == 0
            assert torch.abs(_result.batch["response"] - result.batch["response"]).sum() == 0
            captured = capsys.readouterr()
            assert "Successfully load pre-generated data from" in captured.out
            _result = result
