import pandas as pd
import json
import os

# 配置路径
input_parquet = "./test-00000-of-00030.parquet" # 替换为你的 parquet 文件名
output_jsonl = "./train.jsonl"

# 确保输出目录存在
os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

# 分块读取（防止内存溢出）
print("开始转换 Parquet...")
reader = pd.read_parquet(input_parquet, engine='pyarrow')

with open(output_jsonl, 'w', encoding='utf-8') as f:
    # 假设你的文本在 'content' 列，Megatron 需要 'text' 列
    for text in reader['text']:
        if text: # 过滤空行
            json_record = json.dumps({"text": text}, ensure_ascii=False)
            f.write(json_record + '\n')

print(f"转换完成！明文已保存至: {output_jsonl}")