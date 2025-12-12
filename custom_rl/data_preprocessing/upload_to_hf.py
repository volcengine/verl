"""Upload char_count dataset to Hugging Face Hub"""
import os
from huggingface_hub import HfApi, create_repo
from datasets import Dataset
import pandas as pd

# ===================================
# [INFO] Upload dataset to Hugging Face Hub
# ===================================

def upload_dataset_to_hf(
    data_path: str,
    repo_id: str,
    token: str = None,
    private: bool = True
):
    """
    Upload parquet files to Hugging Face Hub
    
    Args:
        data_path: Path to the data directory containing sft/ and rl/ folders
        repo_id: Repository ID on HF Hub (e.g., "username/dataset-name")
        token: HF token (optional if already logged in)
        private: Whether to make the repo private
    """
    # Initialize HF API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            token=token
        )
        print(f"Created repository: {repo_id}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Upload SFT data
    sft_path = os.path.join(data_path, "sft")
    if os.path.exists(sft_path):
        print("Uploading SFT data...")
        for file in ["train.parquet", "test.parquet"]:
            file_path = os.path.join(sft_path, file)
            if os.path.exists(file_path):
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"sft/{file}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token
                )
                print(f"  Uploaded: sft/{file}")
    
    # Upload RL data
    rl_path = os.path.join(data_path, "rl")
    if os.path.exists(rl_path):
        print("Uploading RL data...")
        for file in ["train.parquet", "test.parquet"]:
            file_path = os.path.join(rl_path, file)
            if os.path.exists(file_path):
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"rl/{file}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token
                )
                print(f"  Uploaded: rl/{file}")
    
    # Create and upload README
    readme_content = """# Char Count Dataset

This dataset is used for training language models on character counting tasks.

## Dataset Structure

```
.
├── sft/
│   ├── train.parquet
│   └── test.parquet
└── rl/
    ├── train.parquet
    └── test.parquet
```

## SFT Data Format
- `prompt`: The question asking to count characters
- `response`: The step-by-step solution with the answer

## RL Data Format
- `prompt`: The question in chat format
- `data_source`: "char_count"
- `ability`: "other"
- `reward_model`: Contains ground truth for reward calculation
- `extra_info`: Additional information including the full response

## Usage

```python
from datasets import load_dataset

# Load SFT data
sft_dataset = load_dataset("YOUR_USERNAME/char-count-dataset", data_files="sft/*.parquet")

# Load RL data
rl_dataset = load_dataset("YOUR_USERNAME/char-count-dataset", data_files="rl/*.parquet")
```
"""
    
    # Upload README
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    print("Uploaded README.md")
    
    print(f"\nDataset successfully uploaded to: https://huggingface.co/datasets/{repo_id}")


def preview_dataset(data_path: str):
    """Preview the dataset before uploading"""
    print("=== Dataset Preview ===\n")
    
    # Preview SFT data
    sft_train = os.path.join(data_path, "sft", "train.parquet")
    if os.path.exists(sft_train):
        df = pd.read_parquet(sft_train)
        print("SFT Train Data:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First example:")
        print(f"    Prompt: {df.iloc[0]['prompt']}")
        print(f"    Response: {df.iloc[0]['response'][:100]}...")
        print()
    
    # Preview RL data
    rl_train = os.path.join(data_path, "rl", "train.parquet")
    if os.path.exists(rl_train):
        df = pd.read_parquet(rl_train)
        print("RL Train Data:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First example prompt: {df.iloc[0]['prompt']}")
        print(f"  First example data_source: {df.iloc[0]['data_source']}")
        print(f"  First example ability: {df.iloc[0]['ability']}")
        print(f"  First example reward_model: {df.iloc[0]['reward_model']}")
        print(f"  First example extra_info: {df.iloc[0]['extra_info']}")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to char_count data directory")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repo ID (e.g., username/dataset-name)")
    parser.add_argument("--token", type=str, default=None, help="HF token (optional if logged in)")
    parser.add_argument("--private", action="store_true", help="Make the dataset private")
    parser.add_argument("--preview", action="store_true", help="Preview dataset without uploading")
    
    args = parser.parse_args()
    
    if args.preview:
        preview_dataset(args.data_path)
    else:
        upload_dataset_to_hf(
            data_path=args.data_path,
            repo_id=args.repo_id,
            token=args.token,
            private=args.private
        ) 