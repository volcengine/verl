export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

BACKEND=vllm
CKPT_PATH="./ckpt/DeepSeek-V3-Base/"

wait_server_ready() {
    server=$1
    ip=$2
    port=$3
    while true; do
        echo "wait $server server ready at $ip:$port..."
        result=`echo -e "\n" | telnet $ip $port 2> /dev/null | grep Connected | wc -l`
        if [ $result -eq 1 ]; then
            break
        else
            sleep 1
        fi
    done
}

ps -ef | grep "python proxy.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

nohup python proxy.py &> proxy.log &

wait_server_ready proxy localhost $PROXY_BACKEND_PORT

echo "teacher proxy is ready"

CUDA_VISIBLE_DEVICES=$i nohup python worker.py --backend $BACKEND --tp-size 1 --seq-len 4096 --ckpt-path $CKPT_PATH &> worker_$i.log &
echo "start teacher worker_$i..."

echo "teacher server is ready"