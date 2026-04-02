import shutil
from pathlib import Path

# 定义源目录和目标目录
SOURCE_DIR = Path("/home/zyp/Desktop/knives_cleaned_aligned")
TARGET_DIR = Path("/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_dataset")

# 确保目标目录存在，如果不存在会自动创建#
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# 定义要添加的四个后缀
suffixes = ["_up", "_down", "_top", "_low"]

print("开始复制并重命名 obj 文件...")

# 遍历 kitchen_knife_1 到 kitchen_knife_135
for i in range(1, 136):
    base_name = f"kitchen_knife_{i}"
    source_obj_path = SOURCE_DIR / f"{base_name}.obj"
    
    # 检查源文件是否存在
    if not source_obj_path.exists():
        print(f"[跳过] 找不到源文件: {source_obj_path}")
        continue
        
    print(f"\n处理: {base_name}.obj")
    
    # 复制并重命名出 4 份
    for suffix in suffixes:
        # 拼接新的文件名，例如 knife1_geo_up.obj
        new_filename = f"{base_name}{suffix}.obj"
        target_obj_path = TARGET_DIR / new_filename
        
        # 执行复制
        shutil.copy(source_obj_path, target_obj_path)
        print(f"  └── [成功] 生成副本: {target_obj_path.name}")

print("\n所有文件复制并重命名完成！")