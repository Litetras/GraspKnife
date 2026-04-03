import os
import shutil
import re

# ==========================================
# 配置区域
# ==========================================
# 1. 刚才那个重命名后的输出文件夹 (renamed_output)
SOURCE_DIR = "/home/zyp/Desktop/hammers" 

# 2. 最终存放精选模型的文件夹
FINAL_DIR = "/home/zyp/Desktop/final_selection"

# 3. 你手动筛选出的 33 个序号
KEEP_INDICES = {
    2, 3, 9, 11, 13, 14, 15, 29, 34, 35, 39, 41, 43, 46, 49, 
    50, 52, 55, 57, 58, 65, 75, 82, 84, 85, 89, 90, 95, 106, 
    111, 113, 121, 141
}

# ==========================================
# 执行提取逻辑
# ==========================================
os.makedirs(FINAL_DIR, exist_ok=True)

print(f"正在从 {SOURCE_DIR} 提取选中的模型...")

def get_file_index(filename):
    """提取文件名中的最后一个数字 (例如 hammer_121.obj -> 121)"""
    match = re.search(r'(\d+)\.obj$', filename)
    return int(match.group(1)) if match else None

found_count = 0
# 遍历源文件夹
for filename in os.listdir(SOURCE_DIR):
    if filename.endswith(".obj"):
        idx = get_file_index(filename)
        
        # 如果序号在你的名单里，就拷贝过去
        if idx in KEEP_INDICES:
            src_path = os.path.join(SOURCE_DIR, filename)
            dst_path = os.path.join(FINAL_DIR, filename)
            
            # 使用 copy2 可以保留文件的时间戳等元数据
            shutil.copy2(src_path, dst_path)
            print(f" [已提取] -> {filename}")
            found_count += 1

print("-" * 60)
print(f"处理完成！")
print(f"名单要求: {len(KEEP_INDICES)} 个")
print(f"成功找到: {found_count} 个")
print(f"精选模型已存放在: {FINAL_DIR}")
print("-" * 60)