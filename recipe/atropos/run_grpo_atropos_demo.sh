#!/bin/bash
# Demo script for GRPO training with Atropos GSM8K environment
# This shows the complete integration with real environment feedback

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GRPO-Atropos Integration Demo${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "verl" ]; then
    echo -e "${RED}Error: Please run this script from the VeRL root directory${NC}"
    exit 1
fi

# Step 1: Check Atropos installation
echo -e "\n${YELLOW}[1/5] Checking Atropos installation...${NC}"
if [ -d "../atropos" ]; then
    echo -e "${GREEN}✓ Atropos found${NC}"
else
    echo -e "${RED}✗ Atropos not found. Please clone it:${NC}"
    echo "cd .. && git clone https://github.com/NousResearch/atropos.git"
    exit 1
fi

# Step 2: Start Atropos GSM8K server
echo -e "\n${YELLOW}[2/5] Starting Atropos GSM8K environment...${NC}"

# Check if server is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Atropos server already running${NC}"
else
    echo "Starting Atropos GSM8K server..."
    cd ../atropos
    python environments/gsm8k_server.py serve --slurm false &
    ATROPOS_PID=$!
    cd ../verl
    
    # Wait for server to start
    echo "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Atropos server started (PID: $ATROPOS_PID)${NC}"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${RED}✗ Failed to start Atropos server${NC}"
        exit 1
    fi
fi

# Step 3: Verify environment status
echo -e "\n${YELLOW}[3/5] Verifying environment status...${NC}"
STATUS=$(curl -s http://localhost:8000/status)
echo "Environment status:"
echo "$STATUS" | python -m json.tool | head -10

# Step 4: Run a test batch
echo -e "\n${YELLOW}[4/5] Testing Atropos integration...${NC}"
python -c "
import sys
sys.path.append('.')
from recipe.atropos.atropos_integration import AtroposEnvironmentClient

client = AtroposEnvironmentClient()
if client.health_check():
    print('✓ Atropos client connected successfully')
    
    # Test getting prompts
    prompts_data = client.get_prompts(batch_size=4)
    if prompts_data:
        print(f'✓ Retrieved {len(prompts_data[\"prompts\"])} prompts')
        print(f'  Example prompt: {prompts_data[\"prompts\"][0][:100]}...')
    else:
        print('✗ Failed to get prompts')
else:
    print('✗ Cannot connect to Atropos')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Integration test failed${NC}"
    exit 1
fi

# Step 5: Launch GRPO training
echo -e "\n${YELLOW}[5/5] Launching GRPO training with real GSM8K feedback...${NC}"
echo -e "${BLUE}Training configuration:${NC}"
echo "  - Model: Qwen/Qwen2-0.5B-Instruct"
echo "  - Environment: GSM8K (7,473 math problems)"
echo "  - Algorithm: GRPO with token-level advantages"
echo "  - Group size: 8 responses per prompt"
echo "  - Real-time correctness evaluation"

# Create output directory
mkdir -p ./checkpoints/grpo_atropos_gsm8k

# Launch training
echo -e "\n${GREEN}Starting training...${NC}"
python recipe/atropos/example_gsm8k_grpo.py

# Cleanup
if [ ! -z "$ATROPOS_PID" ]; then
    echo -e "\n${YELLOW}Stopping Atropos server...${NC}"
    kill $ATROPOS_PID 2>/dev/null || true
fi

echo -e "\n${GREEN}Demo completed!${NC}"