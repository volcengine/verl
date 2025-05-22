import json

# 手动指定的映射关系： (source, level) -> new_level
# 你可以根据需要修改这里
mapping = {
    ("big_math_rl_verified", 1): 1,
    ("big_math_rl_verified", 2): 1,
    ("big_math_rl_verified", 3): 2,
    ("general_reasoning", 1): 2,
    ("general_reasoning", 2): 3,
    ("general_reasoning", 3): 3,
    ("math_dapo", 1): 1,
    ("math_dapo", 2): 1,
    ("math_dapo", 3): 3,
    ("orz_math_data", 1): 3,
    ("orz_math_data", 2): 3,
    ("orz_math_data", 3): 3,
    ("skywork_math", 1): 3,
    ("skywork_math", 2): 3,
    ("skywork_math", 3): 3,
}

input_path = "/home/yangkai/data/data_process/final_merged_math_data.jsonl"
output_path = "/home/yangkai/data/data_process/final_merged_math_data_with_new_level.jsonl"

count_updated = 0
count_skipped = 0

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    
    for line in fin:
        try:
            item = json.loads(line.strip())
            extra = item.get("extra_params", {})
            source = extra.get("source")
            level = extra.get("level")

            key = (source, level)

            if key in mapping:
                new_level = mapping[key]
                item["extra_params"]["new_level"] = new_level
                count_updated += 1
            else:
                count_skipped += 1  # 没有匹配项，跳过添加新字段

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to process line: {e}")
            count_skipped += 1

print(f"Done: update {count_updated} , skip {count_skipped}")
print(f"Output to {output_path}")
