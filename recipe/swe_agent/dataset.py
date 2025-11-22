import json
from datasets import load_dataset

import pandas as pd

cn_beijing_swe_bench="enterprise-public-cn-beijing.cr.volces.com/swe-bench/sweb.eval.x86_64.{project_name}_1776_{instance_number}:latest"
cn_beijing_swe_bench_verified="enterprise-public-cn-beijing.cr.volces.com/swe-bench-verified/sweb.eval.x86_64.{project_name}_1776_{instance_number}:latest"

def generate_swe_bench_image_name(instance_id:str) -> str:
    parts = instance_id.split('__')
    if len(parts) != 2:
        raise ValueError(f"Invalid instance ID format: {instance_id}")
    
    project_name = parts[0].lower()
    instance_number = parts[1].lower()
    return cn_beijing_swe_bench.format(project_name=project_name, instance_number=instance_number)

def generate_swe_bench_verified_image_name(instance_id:str) -> str:
    parts = instance_id.split('__')
    if len(parts) != 2:
        raise ValueError(f"Invalid instance ID format: {instance_id}")
    
    project_name = parts[0].lower()
    instance_number = parts[1].lower()
    return cn_beijing_swe_bench_verified.format(project_name=project_name, instance_number=instance_number)


if __name__ == '__main__':
    # Load datasets - they return DatasetDict objects
    swe_bench_dataset = load_dataset("SWE-bench/SWE-bench")
    swe_bench_verified_dataset = load_dataset("SWE-bench/SWE-bench_Verified")
    
    # 构建JSON结构
    output_data = {
        "swe-bench": {},
        "swe-bench-verified": {}
    }

    
    # Process SWE-bench dataset (use train split or whatever split is available)
    split_name = 'train' if 'train' in swe_bench_dataset else list(swe_bench_dataset.keys())[0]
    for instance in swe_bench_dataset[split_name]:
        output_data["swe-bench"][instance["instance_id"]] = {
            "metadata": instance,
            "image": generate_swe_bench_image_name(instance["instance_id"])
        }

    # Process SWE-bench Verified dataset (use train split or whatever split is available)
    eval, train = [], []
    split_name = 'train' if 'train' in swe_bench_verified_dataset else list(swe_bench_verified_dataset.keys())[0]
    for instance in swe_bench_verified_dataset[split_name]:
        output_data["swe-bench-verified"][instance["instance_id"]] = {
            "metadata": instance,
            "image": generate_swe_bench_verified_image_name(instance["instance_id"])
        }
        item = {
            "prompt": [{"role": "user", "content": "<NOT USED>"}],
            "agent_name": "swe_agent",
            "extra_info": {"tools_kwargs": {
                "dataset_id": "swe-bench-verified", 
                "instance_id": instance["instance_id"],
                "metadata": instance,
                "image": generate_swe_bench_verified_image_name(instance["instance_id"]),
            }},
        }
        eval.append(item)
        train.append(item)
    
    # 写入JSON文件
    with open(f"swe-bench-image.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # 生成数据文件
    train_df = pd.DataFrame(train)
    eval_df = pd.DataFrame(eval)

    train_df.to_parquet("train.parquet")
    eval_df.to_parquet("eval.parquet")