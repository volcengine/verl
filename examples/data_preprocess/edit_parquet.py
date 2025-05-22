import pandas as pd


def process_prompt(prompt):
    content = prompt[0]["content"]
    pos_start = content.find("\n\n")
    pos_end = content.rfind("\n\n")
    if pos_end < pos_start:
        print("error")
    prompt[0]["content"] = content[pos_start+2: pos_end]
    return prompt

# Step 1: 读取
df = pd.read_parquet('/home/share/reasoning/aime-2024.parquet')

# Step 2: 修改
df['prompt'] = df['prompt'].apply(process_prompt)

# Step 1: 按 'name' 字段去重
df_unique = df.drop_duplicates(subset=['prompt'])

# Step 2: 每一行复制8次
df_expanded = pd.concat([df_unique] * 8, ignore_index=True)

# Step 3: 打乱
df_shuffled = df_expanded.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 3: 保存
df_shuffled.to_parquet('/home/share/reasoning/aime-2024-qwen3.parquet', index=False)
