#!/bin/bash

# 实验循环脚本
# OVER_SAMPLE_RATE 从 0.1 到 1.0，每个运行45分钟

cd ~/verl
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 定义 OVER_SAMPLE_RATE 数组
rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

for rate in "${rates[@]}"; do
    echo "=========================================="
    echo "开始实验: OVER_SAMPLE_RATE = $rate"
    echo "时间: $(date)"
    echo "=========================================="
    
    # 设置当前实验的 OVER_SAMPLE_RATE
    export OVER_SAMPLE_RATE=$rate
    
    # 拉取并预处理 gsm8k 数据集
    echo "预处理数据集..."
    python examples/data_preprocess/gsm8k_multiturn_w_tool.py
    
    # 启动训练（后台运行）
    echo "启动训练..."
    bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh &
    TRAIN_PID=$!
    
    # 等待45分钟
    echo "训练将在45分钟后自动终止..."
    sleep 2700  # 45分钟 = 2700秒
    
    # 终止训练进程
    echo "终止训练进程..."
    kill $TRAIN_PID 2>/dev/null
    
    # 清理所有 sglang 相关进程
    echo "清理 sglang 进程..."
    pkill -f sglang
    
    # 等待3分钟
    echo "等待3分钟进行下一组实验..."
    sleep 180  # 3分钟 = 180秒
    
    echo "实验 OVER_SAMPLE_RATE = $rate 完成"
    echo "=========================================="
done

echo "所有实验完成！" 