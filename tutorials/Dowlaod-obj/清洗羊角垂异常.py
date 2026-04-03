import os
import trimesh
import glob

# ==========================================
# 配置区域
# ==========================================
INPUT_DIR = "/home/zyp/Desktop/hammers"             # 你已经下载好的OBJ所在的文件夹
OUTPUT_DIR = "/home/zyp/Desktop/hammers_cleaned"    # 清洗后合格模型存放的新文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 本地模型遍历与【核心形状清洗】
# ==========================================
# 获取所有的 obj 文件路径
obj_files = glob.glob(os.path.join(INPUT_DIR, "*.obj"))
print(f"在输入目录中找到了 {len(obj_files)} 个 OBJ 文件。开始基于几何形状清洗...")

count = 0
dropped_count = 0

for file_path in obj_files:
    filename = os.path.basename(file_path)
    try:
        # 读取本地 OBJ 文件
        scene_or_mesh = trimesh.load(file_path, force='mesh')
        
        # 如果包含多个部件(Scene)，合并为单个网格(Mesh)
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                dropped_count += 1
                continue
            mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
        else:
            mesh = scene_or_mesh

        # 核心预处理：居中到坐标原点 (保持原始尺寸不变)
        mesh.apply_translation(-mesh.centroid)
        
        # ==================================
        # 【核心：羊角锤几何特征清洗】
        # ==================================
        # 提取包围盒三边长：L(长) >= W(宽) >= H(厚)
        extents = sorted(mesh.extents, reverse=True)
        length = extents[0] # 最长边 (手柄总长)
        width = extents[1]  # 中间边 (锤头长度)
        height = extents[2] # 最短边 (手柄/锤头厚度)

        # 异常极小值过滤 (过滤损坏的微小模型)
        if length <= 0.0001:
            dropped_count += 1
            continue

        # 判定 1：必须有长长的手柄。 (排除正方体、球体、头盔等)
        if length < (width * 1.5):
            dropped_count += 1
            continue

        # 判定 2：不能像纸片或刀片一样薄。 (排除贴图平面、极薄的残片)
        if height < (width * 0.15):
             dropped_count += 1
             continue

        # ==================================
        # 【直接导出】(无任何缩放操作)
        # ==================================
        obj_out_path = os.path.join(OUTPUT_DIR, filename)
        mesh.export(obj_out_path)

        # 修复 Blender 导入时的名称乱码问题
        with open(obj_out_path, 'r', encoding='utf-8') as f: 
            lines = f.readlines()
        with open(obj_out_path, 'w', encoding='utf-8') as f:
            clean_name = filename.replace('.obj', '')
            for line in lines:
                if line.startswith('o ') or line.startswith('g '):
                    f.write(f"o {clean_name}\n")
                else: 
                    f.write(line)

        print(f"[{count+1}] 保留形状合格模型: {filename} (比例: L/W={length/width:.1f}, W/H={width/height:.1f})")
        count += 1

    except Exception as e:
        print(f"读取或处理失败跳过: {filename}，原因: {e}")
        dropped_count += 1

# ==========================================
# 清洗报告
# ==========================================
print("="*60)
print("自动清洗汇报：")
print(f"  - 输入总数: {len(obj_files)}")
print(f"  - 成功导出(原尺寸合格锤子): {count} 个")
print(f"  - 因几何特征不符被扔掉的杂物: {dropped_count} 个")
print(f"干净的模型已保存在: {OUTPUT_DIR}")
print("="*60)