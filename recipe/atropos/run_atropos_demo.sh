#!/bin/bash

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Atropos-VERL Integration Demo Runner
# This script runs the production Atropos-VERL integration with real VERL infrastructure

set -e  # Exit on any error

echo "🚀 Atropos-VERL Integration Demo"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the VERL repository root"
    echo "   Current directory: $(pwd)"
    echo "   Expected: VERL repository root with requirements.txt"
    exit 1
fi

# Check Python environment
echo "🔍 Checking Python environment..."
python --version
pip --version

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Check VERL installation
echo "🔍 Checking VERL installation..."
python -c "
try:
    import verl
    print(f'✓ VERL installed: {verl.__version__}')
except ImportError:
    print('❌ VERL not installed. Please install VERL first.')
    exit(1)
"

# Check optional dependencies
echo "🔍 Checking optional dependencies..."
python -c "
optional_deps = {
    'vllm': 'vLLM inference engine',
    'sglang': 'SGLang inference engine'
}

for dep, desc in optional_deps.items():
    try:
        __import__(dep)
        print(f'✓ {desc} available')
    except ImportError:
        print(f'⚠ {desc} not available (optional)')
"

# Run the production demo
echo ""
echo "🎯 Running Atropos-VERL Production Demo..."
echo "=========================================="

# Set environment variables for production
export TOKENIZERS_PARALLELISM="true"
export NCCL_DEBUG="WARN"
export VLLM_LOGGING_LEVEL="WARN"
export VERL_ATROPOS_LOGGING_LEVEL="INFO"

# Run the main demo
python recipe/atropos/main_atropos.py

echo ""
echo "✅ Demo completed successfully!"
echo ""
echo "🎉 Key Features Demonstrated:"
echo "   ✅ VERL inference engines (vLLM/SGLang)"
echo "   ✅ Model loading and training"
echo "   ✅ Complete Atropos API integration"
echo "   ✅ Advantage-weighted SFT loss computation"
echo "   ✅ 3-step RL training loop with policy updates"
echo "   ✅ Memory-efficient inference engine management"
echo "   ✅ Robust error handling for API connectivity"
echo ""
echo "📚 Next Steps:"
echo "   - Run tests: python recipe/atropos/test_atropos_integration.py"
echo "   - Production training: python recipe/atropos/launch_atropos_verl.py --mode training"
echo "   - Check README: recipe/atropos/README.md" 