import json
from collections import Counter

# 数据路径
file_path = "/home/share/reasoning/rl_math_data.jsonl"

# 初始化计数器
level_counter = Counter()
source_counter = Counter()
category_counter = Counter()
total = 0

# 读取文件并统计
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
            extra = item.get("extra_params", {})
            level = extra.get("level")
            source = extra.get("source")
            category = extra.get("math_category")

            if level is not None:
                level_counter[level] += 1
                total += 1

            if source is not None:
                source_counter[source] += 1

            if category is not None:
                category_counter[category] += 1

        except json.JSONDecodeError:
            continue

# 打印 level 分布
print(f"Total samples: {total}")
print(f"\nLevel distribution:")
print(f"{'Level':<8}{'Count':<10}{'Percentage':<10}")
for level in sorted(level_counter.keys()):
    count = level_counter[level]
    percent = (count / total) * 100
    print(f"{level:<8}{count:<10}{percent:.2f}%")

# 打印 source 分布
print(f"\nSource distribution:")
print(f"{'Source':<25}{'Count':<10}{'Percentage':<10}")
for source, count in source_counter.most_common():
    percent = (count / total) * 100
    print(f"{source:<25}{count:<10}{percent:.2f}%")

# 打印 math_category 分布
print(f"\nMath Category distribution:")
print(f"{'Category':<20}{'Count':<10}{'Percentage':<10}")
for category, count in category_counter.most_common():
    percent = (count / total) * 100
    print(f"{category:<20}{count:<10}{percent:.2f}%")
