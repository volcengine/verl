#!/bin/bash
# Qwen3-236B Setup Script for VERL on SLURM with H100 GPUs
# This script helps you prepare everything needed to run Qwen3-236B training

set -euo pipefail

# ================= Configuration =================
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH="$RAY_DATA_HOME/models/Qwen3-235B-A22B"
MCORE_MODEL_PATH="$RAY_DATA_HOME/models/Qwen3-235B-A22B_dist_ckpt_mcore/"
DATASET_DIR="$RAY_DATA_HOME/dataset"

echo "🚀 Setting up Qwen3-236B for VERL training on H100 GPUs"
echo "Data home: $RAY_DATA_HOME"

# ================= Step 1: Create Directories =================
echo "📁 Creating necessary directories..."
mkdir -p "$RAY_DATA_HOME/models"
mkdir -p "$RAY_DATA_HOME/dataset" 
mkdir -p "$RAY_DATA_HOME/ckpt"
mkdir -p "$RAY_DATA_HOME/logs"

# ================= Step 2: Environment Setup =================
echo "Activating verl environment..."
eval "$(conda shell.bash hook)"
conda activate verl

echo "Model path: $MODEL_PATH"
# ================= Step 4: Model Download =================
echo "🤖 Downloading Qwen3-235B-A22B model..."

if [ ! -d "$MODEL_PATH" ]; then
    echo "Downloading Qwen3-235B-A22B model to $MODEL_PATH..."
    # Download the model
    echo "Starting model download (this will take several hours)..."
    hf download Qwen/Qwen3-235B-A22B --local-dir "$MODEL_PATH" --local-dir-use-symlinks False
    
    echo "✅ Model download completed!"
else
    echo "✅ Model already exists at $MODEL_PATH"
fi

# ================= Step 5: Model Conversion =================
echo "🔄 Converting model to Megatron distributed checkpoint format..."

if [ ! -d "$MCORE_MODEL_PATH" ]; then
    echo "Converting Qwen3-235B-A22B to Megatron format..."
    echo "⚠️  This conversion process will take approximately 4 hours"
    echo "⚠️  Make sure you have sufficient disk space (model size × 2)"
    
    read -p "Do you want to proceed with model conversion? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting model conversion..."
        python scripts/converter_hf_to_mcore.py \
            --hf_model_path "$MODEL_PATH" \
            --output_path "$MCORE_MODEL_PATH" \
            --use_cpu_initialization
        echo "✅ Model conversion completed!"
    else
        echo "⚠️  Model conversion skipped. You'll need to run this later."
        echo "Command to run conversion:"
        echo "python scripts/converter_hf_to_mcore.py --hf_model_path $MODEL_PATH --output_path $MCORE_MODEL_PATH --use_cpu_initialization"
    fi
else
    echo "✅ Megatron checkpoint already exists at $MCORE_MODEL_PATH"
fi

# ================= Step 6: Dataset Download =================
echo "📊 Downloading training datasets..."

# Download DAPO math dataset
if [ ! -f "$DATASET_DIR/dapo-math-17k.parquet" ]; then
    echo "Downloading DAPO math dataset..."
    # You may need to adjust this URL or provide instructions for manual download
    echo "Please manually download the DAPO math dataset to $DATASET_DIR/dapo-math-17k.parquet"
    echo "Contact the VERL team for access to this dataset"
fi

# Download AIME 2024 test dataset  
if [ ! -f "$DATASET_DIR/aime-2024.parquet" ]; then
    echo "Downloading AIME 2024 test dataset..."
    echo "Please manually download the AIME 2024 dataset to $DATASET_DIR/aime-2024.parquet"
    echo "Contact the VERL team for access to this dataset"
fi

# ================= Step 7: System Verification =================
echo "🔍 Verifying system requirements..."

# Check CUDA version
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1
    
    echo "GPU information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  nvidia-smi not found. Please ensure CUDA drivers are installed."
fi

# Check Python version
echo "Python version: $(python --version)"

# Check disk space
echo "Available disk space in $RAY_DATA_HOME:"
df -h "$RAY_DATA_HOME"

# ================= Step 8: Configuration Summary =================
echo ""
echo "📋 Setup Summary:"
echo "=================="
echo "Model path: $MODEL_PATH"
echo "Megatron checkpoint path: $MCORE_MODEL_PATH"
echo "Dataset directory: $DATASET_DIR"
echo ""
echo "Resource Requirements for H100 Training:"
echo "- Nodes: 16 (128 H100 GPUs total)"
echo "- Memory per node: ~500GB+ recommended"
echo "- Storage: ~1TB for model + checkpoints"
echo "- Network: High-speed interconnect (InfiniBand recommended)"
echo ""
echo "Next Steps:"
echo "1. Ensure your SLURM cluster has the required resources"
echo "2. Adjust partition names in the SLURM script if needed"
echo "3. Set up Wandb API key for logging (optional)"
echo "4. Submit the job: sbatch run_qwen3_236b_h100_slurm.slurm"
echo ""
echo "✅ Setup script completed!"
echo ""
echo "For issues, check:"
echo "- VERL GitHub: https://github.com/volcengine/verl"
echo "- Documentation: https://verl.readthedocs.io/"
