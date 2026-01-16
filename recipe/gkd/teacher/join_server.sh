export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

PROXY_IP="127.0.0.1"
BACKEND=vllm
CKPT_PATH="/path/to/TEACHER_MODEL/"

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

# pkill -f "python proxy.py"
# pkill -f "python worker.py"
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

wait_server_ready proxy $PROXY_IP $PROXY_BACKEND_PORT

echo "teacher proxy is ready"

nohup python worker.py --backend $BACKEND --proxy-addr $PROXY_IP:$PROXY_BACKEND_PORT --tp-size 8 --n-logprobs 256 --ckpt-path $CKPT_PATH &> worker.log &

echo "teacher server is ready"
