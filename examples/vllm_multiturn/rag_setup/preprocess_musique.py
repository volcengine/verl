import json
import re  
from collections import defaultdict
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi, login


def transform_musique_data(input_path: str, sample_config: dict, hf_repo_id: str, token: str = None) -> None:
    """Transforms Musique data and uploads directly to Hugging Face.

    Reads data, categorizes by detailed hop type, sorts categories by ID,
    selects N samples uniformly spaced from each sorted category,
    combines samples, and uploads to Hugging Face dataset repository.

    Args:
        input_path: Path to the input JSONL file.
        sample_config: Dictionary specifying samples per detailed hop type (e.g., {"2hop": 400, "3hop1": 150, ...}).
        hf_repo_id: Hugging Face repository ID (e.g., "username/dataset-name").
        token: Hugging Face API token. If None, will use the token from huggingface-cli login.
    """
    # Login to Hugging Face if token is provided
    if token:
        login(token=token)
    
    print(f"Reading all data from {input_path} for sampling...")
    all_data = []
    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            for line_num, line in enumerate(infile, 1):
                try:
                    data = json.loads(line)
                    if "id" in data:
                        all_data.append(data)
                    else:
                        print(f"Warning: Skipping line {line_num} due to missing 'id' field in {input_path}")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON in line {line_num} of {input_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except Exception as e:
        print(f"Error reading file {input_path}: {e}")
        return
    print(f"Read {len(all_data)} total samples with IDs.")

    # Detailed Categorization by hop type
    categorized_data = defaultdict(list)
    print("Categorizing data by detailed hop type (e.g., 3hop1, 4hop2)...")
    for data in all_data:
        q_id = data["id"]
        match = re.match(r"^(2hop|3hop[12]|4hop[123])__", q_id)
        if match:
            detailed_hop_type = match.group(1)
            categorized_data[detailed_hop_type].append(data)

    # Deterministic sampling using sorting and uniform index selection
    final_sample_list = []
    total_target = sum(sample_config.values())
    print(f"Sampling deterministically via uniform selection from sorted lists to get {total_target} samples...")
    # Check if all requested hop types exist in config
    for hop_type in sample_config.keys():
        if hop_type not in categorized_data:
            print(f"Warning: Hop type '{hop_type}' requested in config but not found in data.")

    for hop_type, target_count in sample_config.items():
        available_samples = categorized_data.get(hop_type, [])
        current_count = len(available_samples)
        print(f"  {hop_type}: Found {current_count} samples, need {target_count}.")

        if current_count == 0:
            continue

        # Sort the list for this category by ID
        available_samples.sort(key=lambda x: x["id"])

        selected_samples_for_hop = []
        if current_count < target_count:
            print(f"  Warning: Not enough samples for {hop_type}. Taking all {current_count} sorted samples.")
            selected_samples_for_hop = available_samples
        else:
            # Select target_count indices spread uniformly across the available samples
            print(f"  Selecting {target_count} samples uniformly from {current_count}...")
            # Calculate indices using integer interpretation of evenly spaced points
            indices_to_take = [int(i * current_count / target_count) for i in range(target_count)]
            # Ensure uniqueness in case of rounding issues with small numbers
            indices_to_take = sorted(list(set(indices_to_take)))
            # Adjust if rounding resulted in fewer than target_count unique indices
            while len(indices_to_take) < target_count:
                next_idx = indices_to_take[-1] + 1
                if next_idx < current_count and next_idx not in indices_to_take:
                    indices_to_take.append(next_idx)
                else:
                    break

            # Select samples at the calculated indices
            selected_samples_for_hop = [
                available_samples[idx] for idx in indices_to_take[:target_count]
            ]

        final_sample_list.extend(selected_samples_for_hop)

    print(f"Selected {len(final_sample_list)} samples in total.")
    print("Final sample list constructed in order (hop type, then ID).")

    # Process the selected samples into the format we want for our dataset
    print(f"Processing {len(final_sample_list)} selected samples for upload to Hugging Face...")
    processed_data = {
        "id": [],
        "question": [],
        "answer": [],
        "supporting_paragraphs": []
    }
    
    count = 0
    try:
        for data in final_sample_list:
            try:
                supporting_paragraphs = [
                    p["paragraph_text"] for p in data.get("paragraphs", []) if p.get("is_supporting", False)
                ]

                main_answer = data.get("answer", "")
                # aliases = data.get("answer_aliases", [])

                # all_answers = [main_answer] + (aliases if isinstance(aliases, list) else [])
                # valid_answers = [str(ans).strip() for ans in all_answers if ans and str(ans).strip()]
                # unique_valid_answers = list(set(valid_answers))

                # combined_answer_str = " OR ".join(unique_valid_answers)

                processed_data["id"].append(data.get("id"))
                processed_data["question"].append(data.get("question"))
                processed_data["answer"].append(main_answer)
                processed_data["supporting_paragraphs"].append(supporting_paragraphs)
                
                count += 1
            except KeyError as e:
                print(f"Skipping sample due to missing key {e}: {data.get('id')}")
        
        print(f"Successfully processed {count} samples.")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        return

    # Create a Hugging Face Dataset and push to the Hub
    print(f"Creating Dataset object and pushing to Hugging Face Hub at {hf_repo_id}...")
    try:
        # Create a Dataset object
        dataset = Dataset.from_dict(processed_data)
        
        # Push to Hub
        dataset.push_to_hub(
            hf_repo_id, 
            split="train",
            private=False,  # Set to True if you want a private dataset
            commit_message="Upload processed Musique dataset"
        )
        
        print(f"Successfully uploaded dataset to {hf_repo_id}")
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")
        return


if __name__ == "__main__":
    # Define file paths
    RAW_DIR = Path("data/raw")
    
    # Define detailed sampling configuration
    SAMPLING_CONFIG = {
        "2hop": 7000,
        "3hop1": 1500,
        "3hop2": 1500,
        "4hop1": 1000,
        "4hop2": 1000,
        "4hop3": 1000,
    }  # Total = 1000
    
    # Hugging Face repo ID - CHANGE THIS to your username/repo-name
    HF_REPO_ID = "jan-hq/Musique-subset"
    
    # Your Hugging Face API token (optional if already logged in with huggingface-cli)
    # HF_TOKEN = "hf_..."  # Uncomment and add your token if needed
    
    transform_musique_data(
        str(RAW_DIR / "musique_ans_v1.0_train.jsonl"), 
        SAMPLING_CONFIG,
        HF_REPO_ID,
    )

    print("\nMusique dataset transformation and upload to Hugging Face complete.")