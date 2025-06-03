#!/bin/bash

# Check the NODE_RANK environment variable
if [ "$NODE_RANK" = "0" ]; then
    # First node will be Ray head
    ray start --head --dashboard-host=0.0.0.0 --node-ip-address=$(hostname -I | awk '{print $1}') --port 6379

    # Wait for all nodes to start
    # Default to 8 nodes total (including the head node)
    EXPECTED_NODES=${1:-8}
    MAX_RETRIES=60
    RETRY_INTERVAL=10  # seconds

    echo "Waiting for $EXPECTED_NODES total nodes to be ready..."

    for ((i=1; i<=MAX_RETRIES; i++)); do
        # Get the current number of nodes (including the head node)
        # Count lines starting with ' 1 node_' in the Active section
        RAY_STATUS=$(ray status 2>/dev/null)
        NODE_COUNT=$(echo "$RAY_STATUS" | grep -A 20 "Active:" | grep -c " 1 node_")
        
        echo "Attempt $i/$MAX_RETRIES: $NODE_COUNT/$EXPECTED_NODES nodes ready"
        
        if [ "$NODE_COUNT" -ge "$EXPECTED_NODES" ]; then
            echo "All Ray nodes are up and running!"
            break
        fi
        
        echo "Waiting for more nodes to come online... retrying in $RETRY_INTERVAL seconds"
        sleep $RETRY_INTERVAL
    done

    if [ "$NODE_COUNT" -lt "$EXPECTED_NODES" ]; then
        echo "Warning: Timeout waiting for Ray nodes. Only $NODE_COUNT/$EXPECTED_NODES nodes available."
    fi

else
    # wait for the head node to start
    ray start --address $(host $MASTER_ADDR | awk '/has address/ {print $NF}'):6379  --block
fi
