ray stop --force

export RAY_DEDUP_LOGS=0            # 0: disable ray's log folding 1: enable ray's log folding
export HYDRA_FULL_ERROR=1          # display the accurate error stack

ulimit -n 32768
mkdir logs

NNODES=16                          # number of nodes
NPUS_PER_NODE=16                   # the number of npus for each node
export WORLD_SIZE=$(($NNODES*$NPUS_PER_NODE))

RAY_START_PORT=6766
RAY_DASHBOARD_PORT=8260

MASTER_ADDR="IP FOR MASTER NODE"   # modify it to correspond to the IP of the master node
SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"  # modify it to the communication network card of the current node
# obtain the current node IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
export MASTER_PORT=29444
export HCCL_IF_BASE_PORT=64247
export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TASK_QUEUE_ENABLE=2                      # enable level2 optimization of the sent queue of the ascend operator
export HCCL_BUFFSIZE=300                        # the buffer size of HCCL

export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600

export ASCEND_LAUNCH_BLOCKING=0       # debug usage, which seriously affects performance after use, but the error stack is accurate

export VLLM_USE_V1=1                            # use the V1 engine of vLLM
export VLLM_ENABLE_GRAPH_MODE=1                 # enable vLLM graph mode
export HCCL_OP_EXPANSION_MODE=AIV               # enable the communication mode of AIV
export VLLM_ENABLE_MC2=1                        # enable MC2 communication
export VLLM_DP_SIZE=128                         # configure the DP size of vLLM, this is related to the vLLM instance num

# under the configuration of the vLLM log level of INFO, enable this configuration, print the time of prefill and decode
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=0

if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # the master node starts
  ray start --head --port=$RAY_START_PORT --dashboard-host=0.0.0.0 --node-ip-address=$CURRENT_IP --dashboard-port=$RAY_DASHBOARD_PORT --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # determine whether device_count is equal to NNODES
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash ./recipe/r1_ascend/run_deepseekv3_671b_grpo_megatron_npu.sh
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # the child node attempts to register ray with the master node until successful
  while true; do
      # try to connect to the Ray cluster
      ray start --address="$MASTER_ADDR:$RAY_START_PORT" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # check if the connection is successful
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi