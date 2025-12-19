# preprocess_corpus.py
import json
import re
from pathlib import Path

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def preprocess_corpus(input_file, output_file):
    """Preprocess corpus for indexing"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in):
            try:
                doc = json.loads(line)
                
                # Clean title and text
                title = clean_text(doc.get('title', ''))
                text = clean_text(doc.get('text', ''))
                
                # Create combined content for indexing
                content = f'"{title}"\n{text}' if title else text
                
                # Prepare output document
                output_doc = {
                    'id': doc.get('id', f'doc_{line_num}'),
                    'title': title,
                    'text': text,
                    'contents': content
                }
                
                f_out.write(json.dumps(output_doc) + '\n')
                
            except json.JSONDecodeError:
                print(f"Error parsing line {line_num}")
                continue

if __name__ == "__main__":
    preprocess_corpus('data/raw_corpus.jsonl', 'data/corpus/processed_corpus.jsonl')