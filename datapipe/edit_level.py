import json

# 映射关系： (source, level) -> new_level
mapping = {
    ("big_math_rl_verified", 0): 5,
    ("big_math_rl_verified", 1): 5,
    ("big_math_rl_verified", 2): 3,
    ("big_math_rl_verified", 3): 2,
    ("big_math_rl_verified", 4): 2,
    ("big_math_rl_verified", 5): 2,
    ("big_math_rl_verified", 6): 2,
    ("big_math_rl_verified", 7): 2,
    ("big_math_rl_verified", 8): 2,
    ("big_math_rl_verified", 9): 2,
    ("general_reasoning", 1): 12,
    ("general_reasoning", 2): 9,
    ("general_reasoning", 3): 9,
    ("general_reasoning", 4): 8,
    ("general_reasoning", 5): 6,
    ("math_dapo", 0): 13,
    ("math_dapo", 1): 12,
    ("math_dapo", 2): 12,
    ("math_dapo", 3): 12,
    ("math_dapo", 4): 8,
    ("math_dapo", 5): 7,
    ("math_dapo", 6): 7,
    ("math_dapo", 7): 5,
    ("math_dapo", 8): 5,
    ("math_dapo", 9): 5,
}

input_path = "/home/yangkai/data/data_process/raw_merged_math_data.jsonl"
output_path = "/home/yangkai/data/data_process/raw_merged_math_data_new_level.jsonl"

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

            if source in ["big_math_rl_verified", "math_dapo", "general_reasoning"]:
                level = int(float(level) * 10)%10

            key = (source, level)

            if key in mapping:
                new_level = mapping[key]
                item["extra_params"]["mapped_level"] = new_level
                count_updated += 1
            else:
                if source == "skywork_math":
                    item["extra_params"]["mapped_level"] = level
                elif source == "orz_math_data":
                    item["extra_params"]["mapped_level"] = 13
                else:
                    count_skipped += 1
                    print(source, level)

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to process line: {e}")
            count_skipped += 1

print(f"Done: update {count_updated} , skip {count_skipped}")
print(f"Output to {output_path}")
