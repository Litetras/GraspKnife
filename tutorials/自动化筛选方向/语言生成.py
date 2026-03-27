import json

# 固定配置
suffixes = ["up", "down", "top", "low"]
data = {}

# 循环生成 1~135 所有配置
for num in range(1, 136):
    for suffix in suffixes:
        key = f"kitchen_knife_{num}_{suffix}"
        data[key] = {
            "task1": suffix
        }

# 核心修改：直接保存为 JSON 文件，不打印控制台
with open("knife_config.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("文件已生成！名为 knife_config.json，包含全部1-135数据")
