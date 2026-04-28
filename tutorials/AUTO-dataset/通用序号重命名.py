import os
import glob
import re

# ==========================================
# 配置区域 (按需修改这两项即可通用)
# ==========================================
# 1. 指向你存放原始模型的文件夹
WORK_DIR = "/home/zyp/pan1/objaverse_dataset_5/electric_drills_cleaned_aligned"  

# 2. 设定你想要的文件前缀名称 (例如 "kitchen_knife", "claw_hammer", "cup" 等)
PREFIX = "drill"

# 【自动生成】创建一个专门存放重命名后模型的输出文件夹，保护原始数据
OUTPUT_DIR = os.path.join(WORK_DIR, "renamed_output")

# ==========================================
# 执行重命名逻辑
# ==========================================
if not os.path.exists(WORK_DIR):
    raise FileNotFoundError(f"找不到文件夹: {WORK_DIR}")

# 自动创建输出文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 使用正则表达式提取文件名中的数字，进行“真正的自然数字排序”
def extract_number(filename):
    numbers = re.findall(r'\d+', os.path.basename(filename))
    return int(numbers[-1]) if numbers else 0

# 按提取出的数字大小排序，确保 2 排在 10 前面
obj_files = sorted(glob.glob(os.path.join(WORK_DIR, "*.obj")), key=extract_number)

if not obj_files:
    print(f"在 {WORK_DIR} 中没有找到 .obj 文件。")
    exit()

print(f"共找到 {len(obj_files)} 个 OBJ 文件，准备开始以前缀 '{PREFIX}' 执行重命名流水线...")

for index, old_path in enumerate(obj_files, start=1):
    # 动态使用配置好的前缀进行命名
    new_base_name = f"{PREFIX}_{index}"
    new_filename = f"{new_base_name}.obj"
    
    # 将新文件保存到专门的 OUTPUT_DIR，绝不覆盖原文件
    new_path = os.path.join(OUTPUT_DIR, new_filename)
    
    try:
        # 1. 读取旧文件的所有内容
        with open(old_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 2. 写入新文件夹里的新文件，并在写入时修改内部对象名称
        with open(new_path, 'w', encoding='utf-8') as f:
            for line in lines:
                if line.startswith('o ') or line.startswith('g '):
                    f.write(f"o {new_base_name}\n")
                else:
                    f.write(line)
        
        print(f"[{index}/{len(obj_files)}] 转换成功: {os.path.basename(old_path)} -> {new_filename}")
        
    except Exception as e:
        print(f"处理文件 {old_path} 时出错: {e}")

print("="*60)
print(f"全部重命名完成！")
print(f"✅ 安全起见，所有新模型已存放在: {OUTPUT_DIR}")
print("现在你可以把这个新文件夹里的模型一次性拖入 Blender 中了。")
print("="*60)