#!/bin/bash
# Manual Multi-Node Setup Script for verl
# Usage: 
#   Node 1 (head): bash setup_multinode_manual.sh head
#   Node 2 (worker): bash setup_multinode_manual.sh worker <head_node_ip>

set -e

NODE_ROLE=${1:-head}
HEAD_NODE_IP=${2:-}

# Environment setup
conda activate verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1

# Set wandb API key if you have one
# export WANDB_API_KEY="your_wandb_api_key_here"

if [ "$NODE_ROLE" == "head" ]; then
    echo "🚀 Starting HEAD node..."
    
    # Start Ray head node
    ray start --head \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --port=6379 \
        --disable-usage-stats
    
    echo "✅ Head node started!"
    echo "📊 Dashboard available at: http://$(hostname -I | awk '{print $1}'):8265"
    echo "🔗 Worker nodes should connect to: $(hostname -I | awk '{print $1}'):6379"
    
    # Wait for worker nodes to join
    echo "⏳ Waiting for worker nodes to join..."
    while [ $(ray status | grep "Active:" -A 10 | grep "node_" | wc -l) -lt 2 ]; do
        echo "   Nodes connected: $(ray status | grep "Active:" -A 10 | grep "node_" | wc -l)/2"
        sleep 5
    done
    
    echo "✅ All nodes connected! Current cluster status:"
    ray status
    
elif [ "$NODE_ROLE" == "worker" ]; then
    if [ -z "$HEAD_NODE_IP" ]; then
        echo "❌ Error: Please provide head node IP"
        echo "Usage: bash setup_multinode_manual.sh worker <head_node_ip>"
        exit 1
    fi
    
    echo "🔗 Connecting to head node at $HEAD_NODE_IP:6379..."
    
    # Start Ray worker node
    ray start --address="$HEAD_NODE_IP:6379"
    
    echo "✅ Worker node connected!"
    echo "📊 Check cluster status at: http://$HEAD_NODE_IP:8265"
    
else
    echo "❌ Error: Invalid role '$NODE_ROLE'"
    echo "Usage: bash setup_multinode_manual.sh [head|worker] [head_node_ip]"
    exit 1
fi

echo "🎉 Node setup complete!"

