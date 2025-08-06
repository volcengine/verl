#!/bin/bash

# 增强版实验循环脚本
# 支持信号处理、实验恢复、资源监控等功能

set -e  # 遇到错误时退出

# 信号处理函数
cleanup() {
    echo "收到中断信号，正在清理..."
    if [ ! -z "$TRAIN_PID" ] && kill -0 $TRAIN_PID 2>/dev/null; then
        echo "终止训练进程..."
        kill -9 $TRAIN_PID 2>/dev/null
    fi
    pkill -f sglang
    echo "清理完成，退出"
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 创建日志目录
LOG_DIR="~/verl/experiment_logs/$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"

# 实验状态文件
STATE_FILE="$LOG_DIR/experiment_state.txt"

# 检查是否传入了参数
if [ $# -eq 0 ]; then
    # 如果没有传入参数，使用默认值
    rates=(0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4)
    echo "使用默认的 OVER_SAMPLE_RATE 数组: ${rates[*]}"
else
    # 使用传入的参数作为数组
    rates=("$@")
    echo "使用传入的 OVER_SAMPLE_RATE 数组: ${rates[*]}"
fi

# 验证传入的参数是否为有效的数字
for rate in "${rates[@]}"; do
    if ! [[ "$rate" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "错误：传入的参数 '$rate' 不是有效的数字"
        exit 1
    fi
done

# 检查系统资源
check_system_resources() {
    echo "检查系统资源..."
    
    # 检查GPU
    if ! command -v nvidia-smi &> /dev/null; then
        echo "警告: nvidia-smi 不可用"
    else
        echo "GPU 状态:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | head -8
    fi
    
    # 检查内存
    echo "内存使用情况:"
    free -h
    
    # 检查磁盘空间
    echo "磁盘使用情况:"
    df -h ~/verl
}

# 记录实验开始
echo "实验开始时间: $(date)" | tee -a "$LOG_DIR/experiment_summary.log"
echo "实验参数: ${rates[*]}" | tee -a "$LOG_DIR/experiment_summary.log"

# 检查系统资源
check_system_resources | tee -a "$LOG_DIR/experiment_summary.log"

# 无限循环
while true; do
    for rate in "${rates[@]}"; do
    echo "=========================================="
    echo "开始实验: OVER_SAMPLE_RATE = $rate"
    echo "时间: $(date)"
    echo "=========================================="
    
    # 记录实验开始
    echo "$(date): 开始 OVER_SAMPLE_RATE=$rate 的实验" | tee -a "$LOG_DIR/experiment_summary.log"
    echo "$rate" > "$STATE_FILE"
    
    # 启动训练（后台运行）
    echo "启动训练..."
    bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_benchmark.sh $rate > "$LOG_DIR/train_${rate}_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
    TRAIN_PID=$!
    
    # 等待训练完成或超时
    echo "训练将在45分钟后自动终止，或失败时立即终止..."
    for i in {1..60}; do
        # 检查进程是否还在运行
        if ! kill -0 $TRAIN_PID 2>/dev/null; then
            echo "训练进程已结束"
            # 检查退出状态
            wait $TRAIN_PID
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                echo "训练成功完成" | tee -a "$LOG_DIR/experiment_summary.log"
            else
                echo "训练失败，退出码: $EXIT_CODE" | tee -a "$LOG_DIR/experiment_summary.log"
            fi
            break
        fi
        
        # 每5分钟检查一次GPU使用情况
        if [ $((i % 300)) -eq 0 ]; then
            echo "GPU使用情况检查 (第$((i/60))分钟):"
            if command -v nvidia-smi &> /dev/null; then
                nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -8
            fi
        fi
        
        sleep 1
    done
    
    # 如果进程还在运行，强制终止
    if kill -0 $TRAIN_PID 2>/dev/null; then
        echo "训练超时，强制终止进程..." | tee -a "$LOG_DIR/experiment_summary.log"
        kill -9 $TRAIN_PID 2>/dev/null
    fi
    
    # 清理所有 sglang 相关进程
    echo "清理 sglang 进程..."
    pkill -f sglang
    
    # 等待3分钟
    echo "等待3分钟进行下一组实验..."
    sleep 30  # 3分钟 = 180秒
    
    echo "实验 OVER_SAMPLE_RATE = $rate 完成"
    echo "$(date): OVER_SAMPLE_RATE=$rate 实验完成" | tee -a "$LOG_DIR/experiment_summary.log"
    echo "=========================================="
    done
    
    echo "一轮实验完成，开始下一轮..."
    echo "$(date): 完成一轮实验，开始下一轮" | tee -a "$LOG_DIR/experiment_summary.log"
    echo "=========================================="
done 