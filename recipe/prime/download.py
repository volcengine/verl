#!/usr/bin/env python3
"""
Script to download PRIME-RL/Eurus-2-RL-Data dataset and split into train/test parquet files.
"""

import os
import pandas as pd
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        dataset["train"].to_parquet(os.path.join(out_dir, 'train.parquet'))
        logger.info("Saving test.parquet...")
        dataset["validation"].to_parquet(os.path.join(out_dir, 'test.parquet'))
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    download_eurus_dataset("~/data/code")
        