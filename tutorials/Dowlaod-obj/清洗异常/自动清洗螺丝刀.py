import os
import trimesh
import glob

# ==========================================
# 路径配置区域
# ==========================================
BASE_DIR = "/home/zyp/pan1/objaverse_dataset_5"
INPUT_DIR = os.path.join(BASE_DIR, "screwdrivers")
OUTPUT_DIR = os.path.join(BASE_DIR, "screwdrivers_cleaned")

# ==========================================
# 螺丝刀专属几何验证规则
# L(最长边), W(中间边), H(最短边)
# 规则说明：
# 1. (L >= 2.5 * W): 必须是细长形状，长度至少是宽度的 2.5 倍（排除方块、球体）
# 2. (H >= 0.5 * W): 截面必须具有一定的厚度，不能像纸片或刀片一样扁平（排除薄铁片、扳手）
# ==========================================
check_screwdriver = lambda L, W, H: (L >= 2.5 * W) and (H >= 0.5 * W)

# ==========================================
# 核心清洗流程
# ==========================================
if not os.path.exists(INPUT_DIR):
    print(f"⚠️ 找不到输入目录: {INPUT_DIR}，请检查路径。")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)
obj_files = glob.glob(os.path.join(INPUT_DIR, "*.obj"))

print("\n" + "="*60)
print(f"🚀 开始基于几何形状清洗 [螺丝刀 (screwdrivers)] (共 {len(obj_files)} 个文件)")
print("="*60)

count = 0
dropped_count = 0

for file_path in obj_files:
    filename = os.path.basename(file_path)
    try:
        # 1. 读取本地 OBJ 文件
        scene_or_mesh = trimesh.load(file_path, force='mesh')
        
        # 2. 网格合并
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                dropped_count += 1
                continue
            mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
        else:
            mesh = scene_or_mesh

        # 3. 居中到坐标原点
        mesh.apply_translation(-mesh.centroid)
        
        # ==================================
        # 4. 【核心：提取并验证包围盒特征】
        # ==================================
        extents = sorted(mesh.extents, reverse=True)
        length = extents[0] # L: 最长边 (螺丝刀总长)
        width = extents[1]  # W: 中间边 (手柄宽度)
        height = extents[2] # H: 最短边 (手柄厚度)

        # 异常极小值过滤 (过滤损坏的微小点集)
        if length <= 0.0001:
            dropped_count += 1
            continue

        # 使用螺丝刀专属规则判断
        if not check_screwdriver(length, width, height):
            # 调试用：如果想看被扔掉的比例，可以取消下面这行的注释
            # print(f"  -> 丢弃: {filename} (L:W:H = {length/width:.1f}:{width/width:.1f}:{height/width:.1f})")
            dropped_count += 1
            continue

        # ==================================
        # 5. 导出合格模型并修复乱码
        # ==================================
        obj_out_path = os.path.join(OUTPUT_DIR, filename)
        mesh.export(obj_out_path)

        with open(obj_out_path, 'r', encoding='utf-8') as f: 
            lines = f.readlines()
        with open(obj_out_path, 'w', encoding='utf-8') as f:
            clean_name = filename.replace('.obj', '')
            for line in lines:
                if line.startswith('o ') or line.startswith('g '):
                    f.write(f"o {clean_name}\n")
                else: 
                    f.write(line)

        count += 1
        if count % 20 == 0:
            print(f"  -> 已保留 {count} 个合格的螺丝刀模型...")

    except Exception as e:
        dropped_count += 1

print(f"\n📊 [Screwdrivers] 清洗报告：")
print(f"  - 输入总数: {len(obj_files)}")
print(f"  - ✅ 成功保留(原尺寸合格): {count} 个")
print(f"  - ❌ 几何特征不符剔除: {dropped_count} 个")
print(f"  - 干净模型保存至: {OUTPUT_DIR}")
print("\n🎉 螺丝刀清洗完毕！")