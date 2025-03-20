#!/bin/bash

# ANSI 颜色代码
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 隐藏光标
tput civis

# 确保脚本退出时恢复光标显示
trap 'tput cnorm; exit' INT TERM EXIT

# 获取终端大小的函数
get_terminal_size() {
    LINES=$(tput lines)
    COLUMNS=$(tput cols)
}

# 清除从当前行到屏幕底部的内容
clear_screen_below() {
    echo -en "\033[J"
}

# 在指定位置打印文本的函数
print_centered() {
    local text="$1"
    local row="$2"
    
    # 计算文本长度（去除颜色代码）
    local text_length=$(echo -e "$text" | sed 's/\x1b\[[0-9;]*m//g' | wc -c)
    text_length=$((text_length - 1))
    
    # 计算左边距使文本居中
    local padding=$(( (COLUMNS - text_length) / 2 ))
    
    # 移动光标到指定位置
    echo -en "\033[${row};${padding}H"
    # 打印文本
    echo -en "$text"
}

# 初始化屏幕
clear

while true; do
    # 获取最新的终端大小
    get_terminal_size
    
    # 移动光标到顶部
    echo -en "\033[H"
    
    # 获取进程信息
    pid=$(ps aux | grep scripts | grep rl | grep -v grep | tail -n 1 | awk '{print $2}')
    
    if [ ! -z "$pid" ]; then
        # 获取命令行
        cmd=$(ps -p $pid -o cmd= 2>/dev/null)
        
        # 提取最后的文件名
        filename=$(basename "$cmd" 2>/dev/null)
        
        # 计算开始打印的行数（垂直居中）
        center_row=$((LINES / 2 - 2))
        
        # 清除显示区域
        clear_screen_below
        
        # 打印标题和信息
        print_centered "${WHITE}==== Process Monitor ====${NC}" $center_row
        print_centered "${WHITE}PID: $pid${NC}" $((center_row + 1))
                
        # 显示文件名而不是完整命令
        print_centered "${WHITE}Script: $filename${NC}" $((center_row + 3))
                
        # 获取并显示运行时间
        elapsed=$(ps -p $pid -o etime= 2>/dev/null)
        if [ ! -z "$elapsed" ]; then
            print_centered "${WHITE}Running Time: $elapsed${NC}" $((center_row + 4))
        else
            print_centered "${WHITE}Process not found${NC}" $((center_row + 4))
        fi
    else
        # 如果没有找到进程，显示错误信息
        center_row=$((LINES / 2))
        clear_screen_below
        print_centered "${WHITE}No matching process found${NC}" $center_row
    fi
    
    # 刷新输出
    echo -en "\033[u"
    
    # 等待更新
    sleep 1
done