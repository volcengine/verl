import json
from collections import defaultdict
from pathlib import Path

from datasets import Dataset
from huggingface_hub import login


def write_json_with_readable_unicode(outfile, data):
    """Write JSON with readable Unicode characters instead of escape sequences."""
    # Use ensure_ascii=False to prevent escaping non-ASCII characters
    json_string = json.dumps(data, ensure_ascii=False)
    outfile.write(json_string + "\n")

def extract_unique_paragraphs_to_corpus(input_paths: list[str], output_jsonl_path: str) -> None:
    """Extracts unique paragraphs and saves as corpus JSONL with 'id' and 'contents' keys."""
    output_dir = Path(output_jsonl_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    paragraphs_data = defaultdict(set)
    print("Starting paragraph extraction...")

    for file_path in input_paths:
        print(f"Processing file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                for line_num, line in enumerate(infile, 1):
                    try:
                        data = json.loads(line)
                        main_question_id = data.get("id")
                        if not main_question_id:
                            print(f"Warning: Missing 'id' in line {line_num} of {file_path}")
                            continue

                        for p in data.get("paragraphs", []):
                            title = p.get("title", "No Title")
                            text = p.get("paragraph_text", "")
                            content = f"{title}\n{text}".strip()

                            if not content:
                                continue

                            paragraphs_data[content].add(main_question_id)

                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON in line {line_num} of {file_path}")
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num} in {file_path}: {e}")
        except FileNotFoundError:
            print(f"Error: Input file not found: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    print(f"Found {len(paragraphs_data)} unique paragraphs.")

    corpus_data = []
    sorted_content = sorted(paragraphs_data.keys())
    for chunk_id, content in enumerate(sorted_content):
        corpus_data.append({
            "id": str(chunk_id),
            "contents": content
        })

    if not corpus_data:
        print("No paragraphs found to save.")
        return
    
    try:
        with open(output_jsonl_path, "w", encoding="utf-8") as outfile:
            for entry in corpus_data:
                # Use our custom function instead of json.dumps directly
                write_json_with_readable_unicode(outfile, entry)
        print(f"Successfully saved corpus JSONL to {output_jsonl_path}")
    except Exception as e:
        print(f"Error saving corpus JSONL file: {e}")


def push_corpus_to_huggingface(corpus_jsonl_path: str, hf_repo_id: str, token: str = None) -> None:
    """Pushes corpus JSONL to Hugging Face Hub."""
    if token:
        login(token=token)
    
    try:
        corpus_data = {"id": [], "contents": []}
        with open(corpus_jsonl_path, "r", encoding="utf-8") as infile:
            for line in infile:
                entry = json.loads(line)
                corpus_data["id"].append(entry["id"])
                corpus_data["contents"].append(entry["contents"])
        
        dataset = Dataset.from_dict(corpus_data)
        
        dataset.push_to_hub(
            hf_repo_id, 
            private=False,
            commit_message="Upload corpus JSONL dataset"
        )
        
        print(f"Successfully uploaded corpus to {hf_repo_id}")
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")


if __name__ == "__main__":
    RAW_DIR = Path("data/raw")
    PROCESSED_DIR = Path("data/processed")

    input_files = [
        str(RAW_DIR / "musique_ans_v1.0_train.jsonl"),
        str(RAW_DIR / "musique_ans_v1.0_dev.jsonl"),
        str(RAW_DIR / "musique_ans_v1.0_test.jsonl"),
    ]
    
    corpus_jsonl_path = str(PROCESSED_DIR / "corpus.jsonl")
    extract_unique_paragraphs_to_corpus(input_files, corpus_jsonl_path)
    
    # Uncomment to push to Hugging Face
    # HF_REPO_ID = "jan-hq/musique-corpus"
    # push_corpus_to_huggingface(corpus_jsonl_path, HF_REPO_ID)