#!/bin/bash

# Function to print section headers
print_section() {
    echo "============================================="
    echo "$1"
    echo "============================================="
}

# Check CPU information
print_section "CPU Information"
echo "Number of CPU cores: $(nproc)"
echo "CPU model: $(cat /proc/cpuinfo | grep 'model name' | head -n 1 | cut -d ':' -f 2 | xargs)"
echo "CPU frequency: $(cat /proc/cpuinfo | grep 'cpu MHz' | head -n 1 | cut -d ':' -f 2 | xargs) MHz"

# Check Memory information
print_section "Memory Information"
total_mem=$(free -h | grep Mem | awk '{print $2}')
used_mem=$(free -h | grep Mem | awk '{print $3}')
free_mem=$(free -h | grep Mem | awk '{print $4}')
echo "Total Memory: $total_mem"
echo "Used Memory: $used_mem"
echo "Free Memory: $free_mem"

# Check GPU information
print_section "GPU Information"
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
else
    echo "No NVIDIA GPU detected or nvidia-smi not installed"
fi