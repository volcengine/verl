import pyarrow.parquet as pq
from datasets import Dataset

def view_parquet_with_datasets(file_path):
    # Load as HF dataset
    dataset = Dataset.from_parquet(file_path)

    # Print basic information
    print(f"Dataset has {len(dataset)} rows")
    print(f"Features: {dataset.features}")

    # Print random examples
    print("\n=== RANDOM EXAMPLES ===")
    for i in range(3):
        idx = i  # Or use random.randint(0, len(dataset)-1) for truly random samples
        print(f"\nExample {idx}:")
        example = dataset[idx]
        for key, value in example.items():
            if isinstance(value, list) and key == "prompt":
                # Handle prompt format
                print(f"{key}: {value[0]['content']}")
            elif isinstance(value, dict) and key == "reward_model":
                # Handle nested dictionary
                print(f"{key}: {value}")
            elif isinstance(value, str) and len(value) > 100:
                print(f"{key}: {value[:100]}...")
            else:
                print(f"{key}: {value}")

    return dataset

# Example usage
file_path = "~/data/math/sky_work_10k_04_21.parquet"
dataset = view_parquet_with_datasets(file_path)
