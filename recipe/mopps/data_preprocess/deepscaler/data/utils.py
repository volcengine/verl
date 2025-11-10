"""Utility functions for loading and processing datasets.

This module provides functions for loading datasets from JSON files and handling
dataset-related operations in the DeepScaler project.
"""

import json
import os
from typing import Any, Dict, List

from . import Dataset, TrainDataset


def load_dataset(dataset: Dataset) -> List[Dict[str, Any]]:
    """Load a dataset from a JSON file.

    Loads and parses a JSON dataset file based on the provided dataset enum.
    The file path is constructed based on whether it's a training or testing dataset.

    Args:
        dataset: A Dataset enum value specifying which dataset to load.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the dataset records.
            Each dictionary represents one example in the dataset.

    Raises:
        ValueError: If the dataset file cannot be found, contains invalid JSON,
            or encounters other file access errors.

    Example:
        >>> load_dataset(TrainDataset.AIME)
        [{'problem': 'Find x...', 'solution': '42', ...}, ...]
    """
    dataset_name = dataset.value.lower()
    data_dir = "train" if isinstance(dataset, TrainDataset) else "test"

    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(data_dir, f"{dataset_name}.json")
    file_path = os.path.join(current_dir, file_path)

    if not os.path.exists(file_path):
        raise ValueError(f"Dataset file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Error loading dataset: {exc}") from exc


if __name__ == '__main__':
    load_dataset(TrainDataset.NUMINA_OLYMPIAD)