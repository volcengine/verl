#!/bin/bash

# 简化版实验循环脚本
# 每个实验运行60秒，间隔30秒

cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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

# 验证传入的参数是否为有效的数字且在合理范围内
for rate in "${rates[@]}"; do
    # 检查是否为有效数字
    if ! [[ "$rate" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "错误：传入的参数 '$rate' 不是有效的数字"
        exit 1
    fi
    
    # 检查是否在合理范围内 (0-1)
    if (( $(echo "$rate < 0" | bc -l) )) || (( $(echo "$rate > 1" | bc -l) )); then
        echo "错误：传入的参数 '$rate' 超出合理范围 (0-1)"
        exit 1
    fi
done

# 无限循环
while true; do
    for rate in "${rates[@]}"; do
    echo "=========================================="
    echo "开始实验: OVER_SAMPLE_RATE = $rate"
    echo "时间: $(date)"
    echo "=========================================="
    
    # 启动训练（后台运行）
    echo "启动训练..."
    # 使用引号保护参数，确保正确传递
    bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_benchmark.sh "$rate" &
    TRAIN_PID=$!
    
    # 等待训练完成或超时
    echo "训练将在60秒后自动终止..."
    for i in {1..60}; do
        # 检查进程是否还在运行
        if ! kill -0 $TRAIN_PID 2>/dev/null; then
            echo "训练进程已结束"
            break
        fi
        sleep 1
    done
    
    # 终止训练进程
    echo "终止训练进程..."
    kill $TRAIN_PID 2>/dev/null
    
    # 清理所有 sglang 相关进程
    echo "清理 sglang 进程..."
    pkill -f sglang
    
    # 等待30秒
    echo "等待30秒进行下一组实验..."
    sleep 30
    
    echo "实验 OVER_SAMPLE_RATE = $rate 完成"
    echo "=========================================="
    done
    
    echo "一轮实验完成，开始下一轮..."
    echo "=========================================="
done 