cd /workspace/lurui-yun/deep_research/verl/
pip3 install -e .

# update for dashboard
pip install -U "ray[default]"

hostip=$(env | grep MLP_HOST=)
hostip=${hostip#*=}
echo $hostip

export VLLM_ATTENTION_BACKEND=XFORMERS

if [ $hostip  = $MLP_WORKER_0_HOST ]; then
    # 主节点
    ray start --head --node-ip-address=$MLP_WORKER_0_HOST --port=6379 --block &
else
    # 其余节点
    sleep 50
    ray start --address $MLP_WORKER_0_HOST:6379 --block &
fi