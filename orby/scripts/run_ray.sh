#!/bin/bash

# Check the NODE_RANK environment variable
if [ "$NODE_RANK" = "0" ]; then
    # First node will be Ray head
    ray start --head --dashboard-host=0.0.0.0 --node-ip-address=$(hostname -I | awk '{print $1}') --port 6379

else
    # wait for the head node to start
    sleep 60
    ray start --address $(host $MASTER_ADDR | awk '/has address/ {print $NF}'):6379  --block
fi
