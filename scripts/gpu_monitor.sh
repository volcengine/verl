#!/bin/bash

while true; do
  # 使用 allrun 获取各节点的 GPU 状态
  allrun bash -c '
    # 定义 12 种明显区分的颜色（选自 256 色码）：
    # 196：鲜红
    # 208：橙色
    # 226：亮黄
    # 46 ：鲜绿
    # 39 ：清亮青色（替换原先不够清晰的 51）
    # 21 ：深蓝
    # 27 ：亮蓝
    # 129：紫色
    # 201：品红
    # 213：亮粉
    # 141：淡紫
    # 82 ：酸橙绿（与鲜绿互补）
    colors=(196 208 226 46 39 27 129 201 213 141 82)
    
    # 延时输出，避免多节点信息混在一起
    sleep $(echo "$OMPI_COMM_WORLD_RANK * 0.5" | bc)
    
    # 根据当前节点编号取模选择颜色
    idx=$(( OMPI_COMM_WORLD_RANK % ${#colors[@]} ))
    color="\033[38;5;${colors[$idx]}m"
    reset="\033[0m"
    
    # 获取 GPU 状态信息
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used,power.draw,utilization.memory,utilization.gpu --format=csv,noheader,nounits)
    
    # 输出节点编号及 GPU 信息，并在每行后重置颜色
    echo -e "${color}Node $OMPI_COMM_WORLD_RANK:${reset}"
    while IFS= read -r line; do
      echo -e "${color}${line}${reset}"
    done <<< "$gpu_info"
  '
  # 每 10 秒刷新一次状态
  sleep 10
done
