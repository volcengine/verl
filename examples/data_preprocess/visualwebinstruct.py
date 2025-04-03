# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
优化的VisualWebInstruct数据集处理器 - 使用datasets API进行批处理
过滤difficulty为3且image数量在2以内的数据
"""

import os
import time
import random
import datasets
from datasets import Dataset, concatenate_datasets
from multiprocessing import Pool, cpu_count
import tempfile
import shutil
from tqdm import tqdm
import gc
import argparse

from verl.utils.hdfs_io import copy, makedirs

class Timer:
    """简单计时器类"""
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        end_time = time.time()
        print(f"[{self.name}] 耗时: {end_time - self.start_time:.2f}秒")


def process_batch_fn(args):
    """处理单个批次（支持多进程）"""
    batch, batch_indices, is_train, data_source, temp_dir, batch_num = args
    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
    
    results = []
    for example, idx in zip(batch, batch_indices):
        question = example.get('question', '')
        
        images = example.get('images', [])
        
        prompt = question
        if images:
            num_images = len(images)
            if num_images == 1:
                prompt = "<image> " + prompt
            else:
                prompt = " ".join(["<image>"] * num_images) + " " + prompt
        
        # 添加指导说明
        prompt = prompt + ' ' + instruction_following
        
        # 创建处理后的数据结构
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": prompt,
            }],
            "images": images,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example.get('short_answer', '')
            },
            "extra_info": {
                'split': 'train' if is_train else 'test', 
                'index': example.get('idx', idx),
                'answer': example.get('answer', ''),
                'short_answer': example.get('short_answer', ''), 
                "question": question,
                'url': example.get('url', ''),
                'answer_type': example.get('answer_type', ''),
                'old_idx': example.get('old_idx', -1),
                'difficulty': example.get('difficulty', 0)  # 保存difficulty信息
            }
        }
        results.append(data)
    
    # 直接将结果写入临时parquet文件
    try:
        split_name = "train" if is_train else "test"
        output_file = os.path.join(temp_dir, f"{split_name}_part_{batch_num:05d}.parquet")
        
        # 转换为Dataset对象并保存
        batch_dataset = Dataset.from_list(results)
        batch_dataset.to_parquet(output_file)
        
        # 验证文件
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return batch_num, len(results), output_file
        else:
            return batch_num, 0, None
    except Exception as e:
        print(f"批次 {batch_num} 出错: {str(e)}")
        return batch_num, 0, None


def optimized_process_dataset(dataset, output_file, is_train=True, batch_size=500, num_workers=None):
    """
    优化的数据集处理函数 - 使用datasets API进行多进程批处理
    """
    if num_workers is None:
        # 使用CPU核心数的70%，避免过度使用资源
        num_workers = max(1, int(cpu_count() * 0.7))
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="datasets_processing_")
    print(f"创建临时目录: {temp_dir}")
    
    try:
        total_batches = (len(dataset) + batch_size - 1) // batch_size
        print(f"将数据集({len(dataset)}条)拆分为{total_batches}个批次，每批{batch_size}条，使用{num_workers}个工作进程")
        
        # 准备批次参数
        batch_args = []
        for i in range(0, len(dataset), batch_size):
            batch_num = i // batch_size
            end_idx = min(i + batch_size, len(dataset))
            batch_indices = list(range(i, end_idx))
            batch = dataset.select(batch_indices)
            batch_args.append((batch, batch_indices, is_train, data_source, temp_dir, batch_num))
        
        # 多进程处理批次
        batch_results = []
        
        with Timer("批处理阶段"):
            with Pool(processes=num_workers) as pool:
                for result in tqdm(pool.imap_unordered(process_batch_fn, batch_args), total=len(batch_args), desc="处理批次"):
                    batch_num, count, file_path = result
                    if file_path is not None:
                        batch_results.append((file_path, count))
                    
                    # 及时释放内存
                    gc.collect()
        
        # 确保至少有一个有效的批次文件
        if not batch_results:
            print("错误: 没有生成有效的批次文件")
            return 0
        
        # 合并批次文件
        valid_files = [file_path for file_path, _ in batch_results]
        total_rows = sum(count for _, count in batch_results)
        
        with Timer("合并阶段"):
            print(f"合并 {len(valid_files)} 个批次文件，总共 {total_rows} 条数据...")
            
            # 使用datasets API分批加载和合并
            datasets_to_merge = []
            chunk_size = 10  # 每次加载的文件数量
            
            for i in range(0, len(valid_files), chunk_size):
                chunk = valid_files[i:i+chunk_size]
                print(f"加载文件块 {i//chunk_size + 1}/{(len(valid_files) + chunk_size - 1) // chunk_size}...")
                
                for file in chunk:
                    try:
                        ds = datasets.load_dataset('parquet', data_files=file, split='train')
                        datasets_to_merge.append(ds)
                    except Exception as e:
                        print(f"无法加载文件 {file}: {str(e)}")
                
                # 如果累积了足够多的数据集，提前合并以释放内存
                if len(datasets_to_merge) >= 50 or (i + chunk_size >= len(valid_files) and datasets_to_merge):
                    print(f"中间合并 {len(datasets_to_merge)} 个数据集...")
                    merged_ds = concatenate_datasets(datasets_to_merge)
                    datasets_to_merge = [merged_ds]
                    gc.collect()
            
            # 最终合并
            if len(datasets_to_merge) > 1:
                print(f"执行最终合并...")
                final_dataset = concatenate_datasets(datasets_to_merge)
            elif len(datasets_to_merge) == 1:
                final_dataset = datasets_to_merge[0]
            else:
                print("错误: 没有数据集可合并")
                return 0
            
            # 写入最终输出
            print(f"写入合并后的数据集到 {output_file}...")
            final_dataset.to_parquet(output_file)
            
            # 验证输出
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"成功: 输出文件 {output_file} 已创建，大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
            else:
                print(f"错误: 输出文件 {output_file} 未成功创建")
        
        return total_rows
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return 0
    finally:
        # 清理临时目录
        print(f"清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/data/yiming/data/visualwebinstruct_verified/RL_1image')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--dataset_name', default='TIGER-Lab/VisualWebInstruct-Verified')
    parser.add_argument('--batch_size', type=int, default=1000, help='批处理大小 (默认: 1000)')
    parser.add_argument('--num_workers', type=int, default=None, help='并行工作进程数量 (默认: CPU核心数*0.7)')
    parser.add_argument('--test_size', type=int, default=200, help='测试集大小 (默认: 200)')
    
    args = parser.parse_args()
    
    data_source = args.dataset_name
    batch_size = args.batch_size
    num_workers = args.num_workers
    test_size = args.test_size

    with Timer("加载数据集"):
        print(f"加载数据集: {data_source}")
        dataset = datasets.load_dataset(data_source)

        print(f"数据集划分: {dataset.keys()}")
        
        # 获取主要数据
        if 'train' in dataset:
            main_dataset = dataset['train']
        else:
            main_split = list(dataset.keys())[0]
            main_dataset = dataset[main_split]
        
        print(f"总数据量: {len(main_dataset)} 条")
    
    # 添加过滤步骤，只保留difficulty为3且image数量在2以内的样本
    with Timer("过滤数据集"):
        print("过滤数据集: 只保留difficulty为3且image数量在1以内的样本...")
        
        def filter_function(example):
            # 检查difficulty是否为3
            has_correct_difficulty = example.get('difficulty', 0) == 3
            
            # 检查images数量是否≤2
            images = example.get('images', [])
            has_valid_image_count = len(images) <= 1
            
            # 同时满足这两个条件
            return has_correct_difficulty and has_valid_image_count
        
        # 应用过滤器
        filtered_dataset = main_dataset.filter(filter_function)
        
        print(f"原始数据量: {len(main_dataset)} 条")
        print(f"过滤后数据量: {len(filtered_dataset)} 条")
        
        # 确保过滤后的数据集足够大
        if len(filtered_dataset) <= test_size:
            raise ValueError(f"过滤后的数据集太小 ({len(filtered_dataset)} 条)，无法创建 {test_size} 条的测试集")
    
    with Timer("创建数据集划分"):
        # 设置随机种子以确保可复现性
        random.seed(42)
        
        # 随机选择固定数量的样本作为测试集
        all_indices = list(range(len(filtered_dataset)))
        test_indices = random.sample(all_indices, test_size)
        train_indices = [i for i in all_indices if i not in test_indices]
        
        # 创建训练集和测试集
        train_dataset = filtered_dataset.select(train_indices)
        test_dataset = filtered_dataset.select(test_indices)
        
        print(f"创建数据集划分完成: {len(train_dataset)} 条训练数据, {len(test_dataset)} 条测试数据")
        assert len(test_dataset) == test_size, f"测试集应该有 {test_size} 条样本"

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # 创建输出目录
    os.makedirs(local_dir, exist_ok=True)

    # 处理训练集
    with Timer("处理训练集"):
        print(f"\n开始优化处理训练集...")
        train_output = os.path.join(local_dir, 'train.parquet')
        train_count = optimized_process_dataset(
            dataset=train_dataset, 
            output_file=train_output, 
            is_train=True, 
            batch_size=batch_size,
            num_workers=num_workers
        )
    
    # 处理测试集
    with Timer("处理测试集"):
        print(f"\n开始优化处理测试集...")
        test_output = os.path.join(local_dir, 'test.parquet')
        test_count = optimized_process_dataset(
            dataset=test_dataset, 
            output_file=test_output, 
            is_train=False, 
            batch_size=batch_size,
            num_workers=num_workers
        )
    
    print(f"\n处理完成: {train_count} 条训练样本, {test_count} 条测试样本")
    assert test_count == test_size, f"最终测试集大小 ({test_count}) 应该恰好为 {test_size}"

    if hdfs_dir is not None:
        with Timer("复制到HDFS"):
            print(f"复制数据到HDFS: {hdfs_dir}")
            makedirs(hdfs_dir)
            copy(src=local_dir, dst=hdfs_dir)
            print("HDFS复制完成")

    print("所有任务成功完成!")