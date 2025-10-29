import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description='Split DAPO dataset into subsets')
    parser.add_argument('--dataset', type=str, default='open-r1/DAPO-Math-17k-Processed',
                        help='Path to the dataset to load (default: open-r1/DAPO-Math-17k-Processed)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the subset parquet files')
    parser.add_argument('--num_subsets', type=int, default=5,
                        help='Number of subsets to split into (default: 5)')

    args = parser.parse_args()

    # Set split and language based on dataset
    if args.dataset == "open-r1/DAPO-Math-17k-Processed":
        split = 'train'
        language = 'en'
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not supported. Only 'open-r1/DAPO-Math-17k-Processed' is currently supported.")

    # Load dataset
    dataset = load_dataset(args.dataset, language)[split]
    print(f"Loading dataset: {args.dataset}, config: {language}, split: {split}")
    print(f"There are {len(dataset)} samples in total.")

    # Split into N shards and save as Parquet
    for i in range(args.num_subsets):
        subset = dataset.shard(num_shards=args.num_subsets, index=i)
        subset_path = f"{args.output_dir}/subset_{i}.parquet"
        subset.to_parquet(subset_path)
        print(f"Saved subset {i} with {len(subset)} samples to {subset_path}")


if __name__ == "__main__":
    main()