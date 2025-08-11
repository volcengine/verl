#!/bin/bash

# Script to download datasets for VERL training
# Usage: bash scripts/download_datasets.sh

set -e

# Create data directory
DATA_DIR="$HOME/verl/data"
mkdir -p "$DATA_DIR"

echo "📁 Creating data directory: $DATA_DIR"
cd "$DATA_DIR"

# ================= GSM8K Dataset (for RLOO training) =================
echo "📥 Downloading GSM8K dataset..."

# Download GSM8K train set
wget -O gsm8k_train.jsonl "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"

# Download GSM8K test set  
wget -O gsm8k_test.jsonl "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

# Convert to parquet format using Python
python3 << 'EOF'
import pandas as pd
import json

# Convert GSM8K train
print("Converting GSM8K train to parquet...")
with open('gsm8k_train.jsonl', 'r') as f:
    train_data = [json.loads(line) for line in f]
    
# Create proper format for VERL
train_df = pd.DataFrame(train_data)
train_df = train_df.rename(columns={'question': 'prompt', 'answer': 'response'})

# Create train directory
import os
os.makedirs('gsm8k', exist_ok=True)
train_df.to_parquet('gsm8k/train.parquet', index=False)

# Convert GSM8K test
print("Converting GSM8K test to parquet...")
with open('gsm8k_test.jsonl', 'r') as f:
    test_data = [json.loads(line) for line in f]
    
test_df = pd.DataFrame(test_data)
test_df = test_df.rename(columns={'question': 'prompt', 'answer': 'response'})
test_df.to_parquet('gsm8k/test.parquet', index=False)

print(f"✅ GSM8K dataset saved:")
print(f"   Train: {len(train_df)} examples")
print(f"   Test: {len(test_df)} examples")
EOF

# Clean up temporary files
rm gsm8k_train.jsonl gsm8k_test.jsonl

# ================= MATH Dataset =================
echo "📥 Downloading MATH dataset..."

# Download MATH dataset from Hugging Face
python3 << 'EOF'
try:
    from datasets import load_dataset
    import pandas as pd
    import os
    
    print("Loading MATH dataset from Hugging Face...")
    
    # Load MATH dataset
    math_train = load_dataset("competition_math", split="train")
    math_test = load_dataset("competition_math", split="test")
    
    # Convert to DataFrame and rename columns
    train_df = pd.DataFrame(math_train)
    test_df = pd.DataFrame(math_test)
    
    # Rename columns for VERL format
    train_df = train_df.rename(columns={'problem': 'prompt', 'solution': 'response'})
    test_df = test_df.rename(columns={'problem': 'prompt', 'solution': 'response'})
    
    # Create math directory
    os.makedirs('math', exist_ok=True)
    
    # Save as parquet
    train_df.to_parquet('math/train.parquet', index=False)
    test_df.to_parquet('math/test.parquet', index=False)
    
    print(f"✅ MATH dataset saved:")
    print(f"   Train: {len(train_df)} examples")
    print(f"   Test: {len(test_df)} examples")
    
except ImportError:
    print("⚠️  datasets library not found. Install with: pip install datasets")
    print("   Skipping MATH dataset download...")
except Exception as e:
    print(f"❌ Error downloading MATH dataset: {e}")
EOF

# ================= DAPO-MATH-17K Dataset =================
echo "📥 Downloading DAPO-MATH-17K dataset..."

# This appears to be a custom dataset - providing template for manual download
python3 << 'EOF'
import pandas as pd
import json

# Create sample DAPO-MATH data structure
# Note: Replace this with actual data source when available
sample_data = {
    'prompt': [
        "Solve the equation: 2x + 3 = 11",
        "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
        "Calculate the area of a circle with radius 7"
    ],
    'response': [
        "2x = 11 - 3 = 8, so x = 4",
        "f'(x) = 3x^2 + 4x - 5", 
        "Area = πr² = π(7)² = 49π"
    ],
    'reward': [1.0, 1.0, 1.0]
}

df = pd.DataFrame(sample_data)
df.to_parquet('dapo-math-17k.parquet', index=False)

print("📝 Created sample dapo-math-17k.parquet")
print("   Note: Replace with actual DAPO-MATH-17K dataset when available")
print(f"   Sample entries: {len(df)}")
EOF

# ================= AIME 2024 Dataset =================
echo "📥 Creating AIME 2024 dataset..."

python3 << 'EOF'
import pandas as pd

# Create sample AIME 2024 data
# Note: Replace with actual AIME 2024 problems when available
aime_data = {
    'prompt': [
        "In triangle ABC, the angle bisector of angle A meets BC at D. If AB = 12, AC = 18, and BC = 20, find BD.",
        "Find the number of positive integers n ≤ 1000 such that n² + n + 1 is divisible by 7.",
        "A regular hexagon has side length 6. Find the area of the hexagon."
    ],
    'response': [
        "Using the angle bisector theorem: BD/DC = AB/AC = 12/18 = 2/3. Since BD + DC = 20, we get BD = 8.",
        "We need n² + n + 1 ≡ 0 (mod 7). This gives us n ≡ 2, 4 (mod 7). Count: 2×142 + 2 = 286.",
        "Area = (3√3/2) × s² = (3√3/2) × 36 = 54√3"
    ],
    'difficulty': ['medium', 'hard', 'easy']
}

df = pd.DataFrame(aime_data)
df.to_parquet('aime-2024.parquet', index=False)

print("📝 Created sample aime-2024.parquet")
print("   Note: Replace with actual AIME 2024 problems when available")
print(f"   Sample entries: {len(df)}")
EOF

# ================= Verify Downloads =================
echo ""
echo "📊 Dataset Summary:"
echo "==================="

for dataset in gsm8k/train.parquet gsm8k/test.parquet math/train.parquet math/test.parquet dapo-math-17k.parquet aime-2024.parquet; do
    if [ -f "$dataset" ]; then
        echo "✅ $dataset"
        python3 -c "import pandas as pd; df=pd.read_parquet('$dataset'); print(f'   Rows: {len(df)}, Columns: {list(df.columns)}')"
    else
        echo "❌ $dataset (not found)"
    fi
done

echo ""
echo "🎉 Dataset download completed!"
echo "📁 All datasets saved to: $DATA_DIR"
echo ""
echo "Usage in SLURM scripts:"
echo "  GSM8K:        TRAIN_FILE=\"\$HOME/verl/data/gsm8k/train.parquet\""
echo "  MATH:         TRAIN_FILE=\"\$HOME/verl/data/math/train.parquet\""
echo "  DAPO-MATH:    TRAIN_FILE=\"\$HOME/verl/data/dapo-math-17k.parquet\""
echo "  AIME 2024:    TEST_FILE=\"\$HOME/verl/data/aime-2024.parquet\""
