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
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock external libraries before importing the classes under test
# Mock datasets first as it's used directly in the methods
mock_dataset = MagicMock()
mock_dataset_cls = MagicMock()
mock_dataset_cls.from_dict.return_value = mock_dataset

# Mock other libraries via sys.modules
modules_to_mock = {
    "wandb": MagicMock(),
    "swanlab": MagicMock(),
    "mlflow": MagicMock(),
}

# Apply patches using pytest's monkeypatch or unittest.mock.patch
patchers = [
    patch.dict("sys.modules", {name: mock for name, mock in modules_to_mock.items()}),
    # Patch the availability check within transformers
    patch("transformers.utils.import_utils._is_package_available", MagicMock(side_effect=lambda pkg: True if pkg == "datasets" else False)),
    # Patch the Dataset class at source
    patch("datasets.Dataset", mock_dataset_cls)
]

# Start the patches
for p in patchers:
    p.start()

# Now import the classes after mocks are in place
# Assuming the test runner handles PYTHONPATH correctly or runs from the project root.
from verl.utils.tracking import ValidationGenerationsLogger, RolloutLogger


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_data():
    """Provide sample data for logging."""
    # Revert to list of tuples as expected by the code under test
    return [
        ("input1", "output1", 1.0),
        ("input2", "output2", 2.5),
    ]

@pytest.fixture(autouse=True) # Automatically use this fixture for all tests in the module
def reset_dataset_mocks():
    """Reset call counts for shared dataset mocks before each test."""
    mock_dataset_cls.from_dict.reset_mock()
    mock_dataset.to_parquet.reset_mock()
    # If other methods on mock_dataset_cls or mock_dataset are called and asserted, reset them too.

# --- Tests for ValidationGenerationsLogger ---

def test_validation_logger_init_no_dir():
    """Test ValidationGenerationsLogger initialization without data_dir."""
    logger = ValidationGenerationsLogger()
    assert logger.data_dir is None

def test_validation_logger_init_with_dir(temp_dir):
    """Test ValidationGenerationsLogger initialization with data_dir."""
    logger = ValidationGenerationsLogger(data_dir=temp_dir)
    assert logger.data_dir == temp_dir
    assert os.path.exists(temp_dir) # Check if directory is created

def test_validation_logger_log_database_no_dir(sample_data):
    """Test logging to database raises error if data_dir is not set."""
    logger = ValidationGenerationsLogger()
    with pytest.raises(ValueError, match="data_dir must be provided"):
        logger.log(loggers=['database'], samples=sample_data, step=1, epoch=0)

@patch('datasets.Dataset.from_dict', new=mock_dataset_cls.from_dict)
@patch('datasets.Dataset.to_parquet', new=mock_dataset.to_parquet)
def test_validation_logger_log_database(temp_dir, sample_data): # No need to explicitly add reset_dataset_mocks due to autouse=True
    """Test logging validation generations to database."""
    logger = ValidationGenerationsLogger(data_dir=temp_dir)
    step = 10
    epoch = 1
    logger.log(loggers=['database'], samples=sample_data, step=step, epoch=epoch)

    # Verify Dataset.from_dict call
    # Adjust extraction based on tuple structure
    expected_data_dict = {
        "input": [s[0] for s in sample_data],
        "output": [s[1] for s in sample_data],
        "score": [s[2] for s in sample_data],
        "epoch": [epoch] * len(sample_data),
        "step": [step] * len(sample_data)
    }
    mock_dataset_cls.from_dict.assert_called_once_with(expected_data_dict)

    # Verify to_parquet call
    expected_file_path = os.path.join(temp_dir, f"validation_step_{step}.parquet")
    mock_dataset.to_parquet.assert_called_once_with(expected_file_path)

@patch.object(ValidationGenerationsLogger, 'log_generations_to_wandb')
@patch.object(ValidationGenerationsLogger, 'log_generations_to_swanlab')
@patch.object(ValidationGenerationsLogger, 'log_generations_to_mlflow')
@patch.object(ValidationGenerationsLogger, 'log_generations_to_database')
def test_validation_logger_log_dispatch(mock_db, mock_mlflow, mock_swanlab, mock_wandb, sample_data):
    """Test that log calls the correct sub-methods based on loggers list."""
    logger_no_dir = ValidationGenerationsLogger()
    logger_with_dir = ValidationGenerationsLogger(data_dir="dummy_dir") # Need dir for db

    # Test wandb
    logger_no_dir.log(loggers=['wandb'], samples=sample_data, step=1)
    mock_wandb.assert_called_once_with(sample_data, 1)
    mock_wandb.reset_mock()

    # Test swanlab
    logger_no_dir.log(loggers=['swanlab'], samples=sample_data, step=2)
    mock_swanlab.assert_called_once_with(sample_data, 2)
    mock_swanlab.reset_mock()

    # Test mlflow
    logger_no_dir.log(loggers=['mlflow'], samples=sample_data, step=3)
    mock_mlflow.assert_called_once_with(sample_data, 3)
    mock_mlflow.reset_mock()

    # Test database (requires data_dir)
    logger_with_dir.log(loggers=['database'], samples=sample_data, step=4, epoch=0)
    mock_db.assert_called_once_with(sample_data, 4, 0)
    mock_db.reset_mock()

    # Test multiple loggers
    logger_with_dir.log(loggers=['wandb', 'database'], samples=sample_data, step=5, epoch=1)
    mock_wandb.assert_called_once_with(sample_data, 5)
    mock_db.assert_called_once_with(sample_data, 5, 1)


# --- Tests for RolloutLogger ---

def test_rollout_logger_init_no_dir():
    """Test RolloutLogger initialization without data_dir."""
    logger = RolloutLogger()
    assert logger.data_dir is None

def test_rollout_logger_init_with_dir(temp_dir):
    """Test RolloutLogger initialization with data_dir."""
    logger = RolloutLogger(data_dir=temp_dir)
    assert logger.data_dir == temp_dir
    assert os.path.exists(temp_dir) # Check if directory is created

def test_rollout_logger_log_database_no_dir(sample_data):
    """Test logging rollout to database raises error if data_dir is not set."""
    logger = RolloutLogger()
    with pytest.raises(ValueError, match="Data directory must be provided"):
        logger.log(loggers=['database'], samples=sample_data, step=1, epoch=0)

def test_rollout_logger_log_database_no_epoch(temp_dir, sample_data):
    """Test logging rollout to database raises error if epoch is not provided."""
    logger = RolloutLogger(data_dir=temp_dir)
    with pytest.raises(ValueError, match="Epoch number must be provided"):
        logger.log(loggers=['database'], samples=sample_data, step=1, epoch=None)

@patch('datasets.Dataset.from_dict', new=mock_dataset_cls.from_dict)
@patch('datasets.Dataset.to_parquet', new=mock_dataset.to_parquet)
def test_rollout_logger_log_database(temp_dir, sample_data): # No need to explicitly add reset_dataset_mocks due to autouse=True
    """Test logging rollout generations to database."""
    logger = RolloutLogger(data_dir=temp_dir)
    step = 20
    epoch = 2
    logger.log(loggers=['database'], samples=sample_data, step=step, epoch=epoch)

    # Verify Dataset.from_dict call
    # Adjust extraction based on tuple structure
    expected_data_dict = {
        "input": [s[0] for s in sample_data],
        "output": [s[1] for s in sample_data],
        "score": [s[2] for s in sample_data],
        "epoch": [epoch] * len(sample_data),
        "step": [step] * len(sample_data)
    }
    mock_dataset_cls.from_dict.assert_called_once_with(expected_data_dict)

    # Verify to_parquet call
    expected_file_path = os.path.join(temp_dir, f"rollout_step_{step}.parquet")
    mock_dataset.to_parquet.assert_called_once_with(expected_file_path)

@patch.object(RolloutLogger, 'log_generations_to_database')
def test_rollout_logger_log_dispatch(mock_db, sample_data):
    """Test that RolloutLogger.log calls the correct sub-methods."""
    logger_with_dir = RolloutLogger(data_dir="dummy_dir") # Need dir for db

    # Test database
    logger_with_dir.log(loggers=['database'], samples=sample_data, step=4, epoch=0)
    mock_db.assert_called_once_with(sample_data, 4, 0)
    mock_db.reset_mock()

    # Test non-database logger (should do nothing)
    logger_with_dir.log(loggers=['wandb'], samples=sample_data, step=5, epoch=1)
    mock_db.assert_not_called()

# Cleanup patches if they were started globally
@pytest.fixture(autouse=True, scope="module")
def stop_patches():
    yield
    for p in reversed(patchers):
        p.stop()

