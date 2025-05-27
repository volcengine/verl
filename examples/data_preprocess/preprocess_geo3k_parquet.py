import pandas as pd
import numpy as np

def inspect_and_fix_all_data(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)
    
    print(f"Total rows: {len(df)}")
    
    problematic_rows = []
    
    for idx, row in df.iterrows():
        prompt = row['prompt']
        
        # Convert numpy array to list
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
            df.at[idx, 'prompt'] = prompt
        
        # Check for list content in messages
        if isinstance(prompt, list):
            for msg_idx, msg in enumerate(prompt):
                if isinstance(msg, dict) and 'content' in msg:
                    if isinstance(msg['content'], list):
                        print(f"Row {idx}: Found list content: {msg['content']}")
                        # Fix it
                        msg['content'] = ' '.join(str(x) for x in msg['content'])
                        problematic_rows.append(idx)
    
    print(f"Fixed {len(problematic_rows)} problematic rows")
    
    # Save the completely fixed data
    df.to_parquet(output_file)
    print(f"Saved to {output_file}")
    
    return df

# Fix both files thoroughly
train_df = inspect_and_fix_all_data('~/data/geo3k/train.parquet', '~/data/geo3k/train_clean.parquet')
test_df = inspect_and_fix_all_data('~/data/geo3k/test.parquet', '~/data/geo3k/test_clean.parquet')

# Verify the fix worked
print("\n=== Verification ===")
for split, df in [('train', train_df), ('test', test_df)]:
    print(f"\n{split} data:")
    
    # Check data types
    numpy_count = df['prompt'].apply(lambda x: isinstance(x, np.ndarray)).sum()
    list_count = df['prompt'].apply(lambda x: isinstance(x, list)).sum()
    print(f"  Numpy arrays: {numpy_count}")
    print(f"  Lists: {list_count}")
    
    # Check for list content
    def has_list_content(row):
        prompt = row['prompt']
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg.get('content'), list):
                    return True
        return False
    
    list_content_count = df[df.apply(has_list_content, axis=1)].shape[0]
    print(f"  Rows with list content: {list_content_count}")

print("\nAll done! Use train_clean.parquet and test_clean.parquet")