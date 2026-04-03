import json

# 定义固定后缀
suffixes = ['up', 'down', 'top', 'low']
# 创建空字典存储键值对
file_map = {}

# 循环生成1~135的所有映射关系
for num in range(1, 30):
    for suffix in suffixes:
        # 键：obj文件名
        obj_file = f"hammer_{num}_{suffix}.obj"#############################
        # 值：对应的grasps.json文件名
        json_file = f"hammer_{num}_{suffix}_grasps.json"####################################
        file_map[obj_file] = json_file

# 格式化输出JSON（带缩进，美观易复制）
print(json.dumps(file_map, indent=2, ensure_ascii=False))
