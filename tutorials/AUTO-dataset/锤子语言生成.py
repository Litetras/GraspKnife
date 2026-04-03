import json

# 固定配置
suffixes = ["up", "down", "top", "low"]
data = {}

# 循环生成 1~135 所有配置
for num in range(1, 30):
    for suffix in suffixes:
        key = f"hammer_{num}_{suffix}"
        data[key] = {
            "task1": suffix
        }

# 核心修改：直接保存为 JSON 文件，不打印控制台
with open("hammer_config.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("文件已生成！名为 hammer_config.json，包含全部1-29数据")
