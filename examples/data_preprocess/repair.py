#!/usr/bin/env python3
"""
改进版诊断和修复Parquet文件的脚本
用法: python repair_improved.py --base_dir /path/to/dir --output_file /path/to/output.parquet
"""

import os
import glob
import argparse
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import sys
from tqdm.auto import tqdm

def check_file(file_path):
    """检查Parquet文件是否有效并返回行数"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return 0
        
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        print(f"文件大小为0字节: {file_path}")
        return 0
        
    try:
        # 使用pyarrow直接读取
        table = pq.read_table(file_path)
        row_count = table.num_rows
        print(f"文件有效: {file_path} - 大小: {file_size} 字节, 行数: {row_count}")
        return row_count
    except Exception as e:
        print(f"无法读取文件 {file_path}: {str(e)}")
        return 0

def find_temp_dirs(base_dir):
    """查找所有temp_开头的目录"""
    temp_dirs = []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and item.startswith('temp_'):
            temp_dirs.append(full_path)
    return temp_dirs

def repair_parquet_files(base_dir, output_file, split_type="train"):
    """查找、验证并合并Parquet文件"""
    # 查找所有temp目录
    temp_dirs = find_temp_dirs(base_dir)
    
    if not temp_dirs:
        print(f"没有找到temp_开头的目录在: {base_dir}")
        # 尝试直接在base_dir中查找文件
        file_pattern = os.path.join(base_dir, f"{split_type}_part_*.parquet")
        files = sorted(glob.glob(file_pattern))
    else:
        print(f"找到 {len(temp_dirs)} 个临时目录:")
        for d in temp_dirs:
            print(f"  - {d}")
        
        # 在所有temp目录中查找文件
        files = []
        for temp_dir in temp_dirs:
            pattern = os.path.join(temp_dir, f"{split_type}_part_*.parquet")
            matched_files = glob.glob(pattern)
            if matched_files:
                print(f"在 {temp_dir} 中找到 {len(matched_files)} 个文件")
                files.extend(matched_files)
    
    if not files:
        print(f"没有找到匹配的文件!")
        return False
    
    print(f"找到 {len(files)} 个Parquet文件")
    
    # 验证文件
    valid_files = []
    total_rows = 0
    
    for file in files:
        rows = check_file(file)
        if rows > 0:
            valid_files.append(file)
            total_rows += rows
    
    if not valid_files:
        print("没有有效的文件可以合并")
        return False
    
    print(f"\n合并 {len(valid_files)}/{len(files)} 个有效文件，总行数: {total_rows}")
    
    # 合并文件
    try:
        # 方法1: 使用pandas
        all_dfs = []
        for file in tqdm(valid_files, desc="读取文件"):
            df = pd.read_parquet(file)
            all_dfs.append(df)
        
        if not all_dfs:
            print("没有有效的DataFrame可以合并")
            return False
            
        print("合并DataFrames...")
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        print(f"写入合并后的数据到 {output_file} (行数: {len(merged_df)})")
        merged_df.to_parquet(output_file, index=False)
        
        # 验证结果
        result_size = os.path.getsize(output_file)
        print(f"合并完成，输出文件大小: {result_size} 字节")
        
        if result_size > 0:
            print("修复成功!")
            return True
        else:
            print("修复失败: 输出文件大小为0")
            return False
            
    except Exception as e:
        print(f"合并过程中出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="诊断和修复Parquet文件")
    parser.add_argument("--base_dir", required=True, help="包含temp_*目录的基础目录")
    parser.add_argument("--output_file", required=True, help="输出的合并Parquet文件路径")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="数据集划分类型")
    
    args = parser.parse_args()
    
    # 显示系统信息
    print(f"Python版本: {sys.version}")
    print(f"Pandas版本: {pd.__version__}")
    print(f"PyArrow版本: {pa.__version__}")
    print(f"基础目录: {args.base_dir}")
    print(f"输出文件: {args.output_file}")
    print(f"数据集划分: {args.split}")
    
    # 检查输入目录
    if not os.path.exists(args.base_dir):
        print(f"错误: 基础目录不存在: {args.base_dir}")
        return 1
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 修复文件
    success = repair_parquet_files(args.base_dir, args.output_file, args.split)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())