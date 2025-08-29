# Copyright 2025 prime team and/or its affiliates
# Copyright 2025 Bytedance Ltd. and/or its affiliates

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

#!/usr/bin/env python3
"""
Script to download PRIME-RL/Eurus-2-RL-Data dataset and split into train/test parquet files.
"""

import logging
import os

from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_eurus_dataset(out_dir):
    """
    Download the PRIME-RL/Eurus-2-RL-Data dataset and split into train/test sets.

    Args:
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random state for reproducible splitting
    """
    try:
        # Download the dataset
        logger.info("Loading PRIME-RL/Eurus-2-RL-Data dataset...")
        dataset = load_dataset("PRIME-RL/Eurus-2-RL-Data", trust_remote_code=True)

        # Save as parquet files
        logger.info("Saving train.parquet...")
        dataset["train"].to_parquet(os.path.join(out_dir, "train.parquet"))
        logger.info("Saving test.parquet...")
        dataset["validation"].to_parquet(os.path.join(out_dir, "test.parquet"))

    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise


if __name__ == "__main__":
    download_eurus_dataset("~/data/code")
