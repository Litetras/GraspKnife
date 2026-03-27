import os
import glob

# ==========================================
# 配置区域
# ==========================================
# 指向你刚才清洗后存放模型的文件夹
WORK_DIR = "/home/zyp/Desktop/knives_cleaned_aligned"

# ==========================================
# 执行重命名逻辑
# ==========================================
if not os.path.exists(WORK_DIR):
    raise FileNotFoundError(f"找不到文件夹: {WORK_DIR}")

# 获取目录下所有的 .obj 文件，并按字母顺序排个序
obj_files = sorted(glob.glob(os.path.join(WORK_DIR, "*.obj")))

if not obj_files:
    print(f"在 {WORK_DIR} 中没有找到 .obj 文件。")
    exit()

print(f"共找到 {len(obj_files)} 个 OBJ 文件，准备开始重命名流水线...")

for index, old_path in enumerate(obj_files, start=1):
    # 生成新的基础名和文件名
    new_base_name = f"kitchen_knife_{index}"
    new_filename = f"{new_base_name}.obj"
    new_path = os.path.join(WORK_DIR, new_filename)
    
    # 极小概率的防碰撞保护：如果本来就叫这个名字，直接跳过
    if old_path == new_path:
        continue
        
    try:
        # 1. 读取旧文件的所有内容
        with open(old_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 2. 写入新文件，并在写入时修改内部对象名称
        with open(new_path, 'w', encoding='utf-8') as f:
            for line in lines:
                # 凡是声明对象(o)或组(g)的地方，强制改成新的基础名
                if line.startswith('o ') or line.startswith('g '):
                    f.write(f"o {new_base_name}\n")
                else:
                    f.write(line)
                    
        # 3. 删除旧文件
        os.remove(old_path)
        
        print(f"[{index}/{len(obj_files)}] 转换成功: {os.path.basename(old_path)} -> {new_filename}")
        
    except Exception as e:
        print(f"处理文件 {old_path} 时出错: {e}")

print("="*60)
print(f"全部重命名完成！")
print(f"现在你可以把 {WORK_DIR} 里的所有模型全选，一次性拖入 Blender 中了。")
print("它们在大纲视图里的名字将是干净整齐的 kitchen_knife_1 到 kitchen_knife_xxx！")
print("="*60)