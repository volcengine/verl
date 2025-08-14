#!/bin/bash
# DeepSeek-V3 671B Setup Script for VERL on SLURM with H100 GPUs
# This script helps you prepare everything needed to run DeepSeek-V3 671B training

set -euo pipefail

VERL_WORKDIR="$HOME/verl"
DATASET_DIR="$VERL_WORKDIR/data"                                    # Dataset directory

# ================= Configuration =================
# ⭐ CUSTOMIZABLE PATHS - Change these to your preferred locations ⭐
BASE_DATA_DIR="/data/xiangmin/dspk"                                           # Main data directory
MODEL_CONFIG_PATH="$BASE_DATA_DIR/DeepSeek-V3-config"               # DeepSeek config files
MCORE_MODEL_PATH="$BASE_DATA_DIR/dpsk-v3-671B-BF16-dist_ckpt"       # Distributed checkpoint
CKPTS_DIR="$BASE_DATA_DIR/ckpts"                                     # Checkpoints directory

echo "🚀 Setting up DeepSeek-V3 671B for VERL training on H100 GPUs"
echo "Base data directory: $BASE_DATA_DIR"
echo "Model config path: $MODEL_CONFIG_PATH"
echo "Megatron checkpoint path: $MCORE_MODEL_PATH"

# ================= Step 1: Create Directories =================
echo "📁 Creating necessary directories..."
mkdir -p "$VERL_WORKDIR/logs"

# ================= Step 2: Download DeepSeek Configuration =================
echo "⚙️  Downloading DeepSeek-V3 configuration files..."

if [ ! -f "$MODEL_CONFIG_PATH/configuration_deepseek.py" ] || [ ! -f "$MODEL_CONFIG_PATH/config.json" ]; then
    echo "Downloading DeepSeek-V3 configuration files..."
    
    # Check if user has HuggingFace CLI
    if ! command -v huggingface-cli &> /dev/null; then
        echo "Installing HuggingFace CLI..."
        pip install huggingface_hub[cli]
    fi
    
    # Download only the configuration files
    echo "Downloading configuration files to $MODEL_CONFIG_PATH..."
    huggingface-cli download deepseek-ai/DeepSeek-V3-0324 \
        configuration_deepseek.py config.json \
        --local-dir "$MODEL_CONFIG_PATH" \
        --local-dir-use-symlinks False
    
    echo "✅ Configuration files downloaded!"
    
    # Important: Modify config.json to remove quantization_config and set MTP to 0
    echo "🔧 Modifying config.json..."
    if [ -f "$MODEL_CONFIG_PATH/config.json" ]; then
        # Create backup
        cp "$MODEL_CONFIG_PATH/config.json" "$MODEL_CONFIG_PATH/config.json.backup"
        
        # Use Python to modify the JSON file
        python3 -c "
import json
import sys

config_path = '$MODEL_CONFIG_PATH/config.json'
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Remove quantization_config if it exists
    if 'quantization_config' in config:
        del config['quantization_config']
        print('Removed quantization_config from config.json')
    
    # Set num_nextn_predict_layers to 0 to disable MTP
    if 'num_nextn_predict_layers' in config:
        config['num_nextn_predict_layers'] = 0
        print('Set num_nextn_predict_layers=0 to disable MTP')
    
    # Write back the modified config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print('✅ config.json modified successfully')
except Exception as e:
    print(f'❌ Error modifying config.json: {e}')
    sys.exit(1)
"
    fi
else
    echo "✅ Configuration files already exist at $MODEL_CONFIG_PATH"
fi

# ================= Step 3: Download Distributed Checkpoint =================
echo "💾 Checking for DeepSeek-V3 distributed checkpoint..."

if [ ! -d "$MCORE_MODEL_PATH" ] || [ -z "$(ls -A $MCORE_MODEL_PATH 2>/dev/null)" ]; then
    echo "⚠️  Distributed checkpoint not found at $MCORE_MODEL_PATH"
    echo ""
    echo "You need to download the distributed checkpoint manually from:"
    echo "https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt/tree/main"
    echo ""
    echo "To download using git-lfs:"
    echo "  git lfs install"
    echo "  git clone https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt $MCORE_MODEL_PATH"
    echo ""
    echo "Or using huggingface-cli:"
    echo "  huggingface-cli download BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt --local-dir $MCORE_MODEL_PATH"
    echo ""
    echo "⚠️  This is a very large download (~1TB). Ensure you have sufficient disk space."
    echo ""
    
    read -p "Do you want to download the checkpoint now using huggingface-cli? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting checkpoint download..."
        echo "⚠️  This will take several hours and requires ~1TB of disk space"
        
        huggingface-cli download BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt \
            --local-dir "$MCORE_MODEL_PATH" \
            --local-dir-use-symlinks False
        
        echo "✅ Checkpoint download completed!"
    else
        echo "⚠️  Checkpoint download skipped. You'll need to download it manually before running training."
    fi
else
    echo "✅ Distributed checkpoint already exists at $MCORE_MODEL_PATH"
fi

# ================= Step 4: Dataset Setup =================
echo "📊 Setting up datasets..."

# Check for DAPO math dataset
if [ ! -f "$DATASET_DIR/dapo-math-17k.parquet" ]; then
    echo "⚠️  DAPO math dataset not found at $DATASET_DIR/dapo-math-17k.parquet"
    echo "Please obtain this dataset and place it at the above path"
fi

# Check for AIME 2024 test dataset  
if [ ! -f "$DATASET_DIR/aime-2024.parquet" ]; then
    echo "⚠️  AIME 2024 dataset not found at $DATASET_DIR/aime-2024.parquet"
    echo "Please obtain this dataset and place it at the above path"
fi

echo "Contact the VERL team for access to the DAPO datasets if needed"

# ================= Step 5: System Verification =================
echo "🔍 Verifying system requirements..."

# Check CUDA version
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA Driver version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1
    
    echo "GPU information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    
    # Check if we have H100 or A100 GPUs
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | grep -E "(H100|A100)" | wc -l)
    if [ $gpu_count -gt 0 ]; then
        echo "✅ Found $gpu_count compatible GPU(s) for DeepSeek-V3 671B"
    else
        echo "⚠️  No H100/A100 GPUs detected. DeepSeek-V3 671B requires high-memory GPUs"
    fi
else
    echo "⚠️  nvidia-smi not found. Please ensure CUDA drivers are installed."
fi

# Check Python version
echo "Python version: $(python --version)"

# Check if verl environment exists
if conda env list | grep -q "verl"; then
    echo "✅ Found verl conda environment"
else
    echo "⚠️  verl conda environment not found. Please set it up first."
fi

# Check disk space
echo "Available disk space in $BASE_DATA_DIR:"
df -h "$BASE_DATA_DIR" 2>/dev/null || df -h "$(dirname "$BASE_DATA_DIR")"

# ================= Step 6: Resource Requirements Summary =================
echo ""
echo "📋 DeepSeek-V3 671B Resource Requirements Summary:"
echo "=================================================="
echo "Model size: ~671B parameters"
echo "Recommended setup: 64 nodes × 8 H100 GPUs = 512 GPUs total"
echo "Memory per node: ~1TB+ recommended"
echo "Storage requirements:"
echo "  - Model checkpoint: ~1TB"
echo "  - Training checkpoints: ~500GB+"
echo "  - Datasets: ~50GB"
echo "  - Total: ~1.5TB+"
echo ""
echo "Parallelism configuration:"
echo "  - Generation TP: 32 (tensor parallelism for vLLM inference)"
echo "  - Training TP: 1 (no tensor parallelism for training)"
echo "  - Training EP: 32 (expert parallelism for MoE)"
echo "  - Training PP: 16 (pipeline parallelism)"
echo ""
echo "Key differences from Qwen3-236B:"
echo "  - More GPUs needed (512 vs 128)"
echo "  - Longer response length (8K vs 4K tokens)"
echo "  - More responses per prompt (16 vs 4)"
echo "  - Expert parallelism enabled for MoE architecture"
echo ""

# ================= Step 7: File Structure Summary =================
echo "📁 Expected file structure:"
echo "$BASE_DATA_DIR/"
echo "├── DeepSeek-V3-config/"
echo "│   ├── configuration_deepseek.py"
echo "│   └── config.json"
echo "├── dpsk-v3-671B-BF16-dist_ckpt/"
echo "│   └── [distributed checkpoint files]"
echo "├── data/"
echo "│   ├── dapo-math-17k.parquet"
echo "│   └── aime-2024.parquet"
echo "├── ckpts/"
echo "│   └── [training checkpoints will be saved here]"
echo "└── logs/"
echo "    └── [training logs will be saved here]"
echo ""

# ================= Step 8: Next Steps =================
echo "🎯 Next Steps:"
echo "=============="
echo "1. Ensure your SLURM cluster has 64+ nodes with H100 GPUs"
echo "2. Download the distributed checkpoint if not done (~1TB)"
echo "3. Obtain and place the DAPO datasets in $DATASET_DIR/"
echo "4. Adjust SLURM partition and module loads in the job script"
echo "5. Set up Wandb API key for logging (optional)"
echo "6. Submit the job: sbatch run_deepseek_671b_h100_slurm.slurm"
echo ""
echo "⚠️  Important Notes:"
echo "- DeepSeek-V3 671B requires significantly more resources than Qwen3-236B"
echo "- The model uses MoE (Mixture of Experts) architecture"
echo "- Training will take longer due to the model size"
echo "- Monitor GPU memory usage and adjust batch sizes if needed"
echo ""
echo "✅ Setup script completed!"
echo ""
echo "For issues, check:"
echo "- VERL GitHub: https://github.com/volcengine/verl"
echo "- DeepSeek documentation: https://github.com/deepseek-ai/DeepSeek-V3"
