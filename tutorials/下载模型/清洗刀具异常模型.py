import os
import trimesh
import json
import random

# ==========================================
# 配置区域
# ==========================================
# 这里使用你已经下载好的那个包含 965 个 UID 的 JSON 文件
UID_LIST_FILE = "kitchen_knife_pure_uids.json" 
OUTPUT_DIR = "/home/zyp/Desktop/knives_cleaned"  # 保存清洗后的模型
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义随机目标尺寸区间 (米)
TARGET_MIN_SCALE = 0.15  
TARGET_MAX_SCALE = 0.35  

# ==========================================
#加载模型缓存
# ==========================================
import objaverse
os.environ["http_proxy"] = "http://127.0.0.1:20171"
os.environ["https_proxy"] = "http://127.0.0.1:20171"

with open(UID_LIST_FILE, 'r', encoding='utf-8') as f:
    knife_uids = json.load(f)

print(f"检查并加载本地缓存...")
# 这步会瞬间跳过已下载的模型
objects = objaverse.load_objects(uids=knife_uids, download_processes=8)

# ==========================================
# 转换与【核心清洗】
# ==========================================
print(f"开始自动几何清洗与缩放...")

count = 0
dropped_count = 0

for uid, glb_path in objects.items():
    try:
        scene_or_mesh = trimesh.load(glb_path, force='mesh')
        
        # 合并 Scene
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                continue
            mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
        else:
            mesh = scene_or_mesh

        # 核心预处理：居中
        mesh.apply_translation(-mesh.centroid)
        
        # ==================================
        # 【新增：几何特征清洗】
        # ==================================
        # 获取包围盒的三个维度的长度：e1 >= e2 >= e3
        extents = sorted(mesh.extents, reverse=True)
        length = extents[0] # 最长边
        width = extents[1]  # 中间边
        height = extents[2] # 最短边

        # 如果最长边特别小，可能不是模型，跳过
        if length <= 0.0001:
            dropped_count += 1
            continue

        # 厨刀形态判定：
        # 1. 不能太“方正”：最长边必须明显大于中间边（比如至少是1.5倍）
        #    排除像碗、正方体、杂乱场景等。
        if length < (width * 1.5):
            # print(f"  -> 跳过 (太方正): {uid[:8]}")
            dropped_count += 1
            continue

        # 2. 不能太“厚”：中间边必须明显大于最短边（比如至少是3倍）
        #    排除像圆柱体、粗棒子等。对于厨刀来说，height通常非常小（刀刃厚度）。
        if width < (height * 3.0):
             # print(f"  -> 跳过 (太厚实): {uid[:8]}")
             dropped_count += 1
             continue

        # ==================================
        # 【随机缩放逻辑】
        # ==================================
        random_target_scale = random.uniform(TARGET_MIN_SCALE, TARGET_MAX_SCALE)
        scale_factor = random_target_scale / length
        mesh.apply_scale(scale_factor)

        # ==================================
        # 【导出 OBJ (保留Blender修复)】
        # ==================================
        obj_filename = f"kitchen_knife_{uid[:8]}.obj"
        obj_out_path = os.path.join(OUTPUT_DIR, obj_filename)
        mesh.export(obj_out_path)

        # 修复 Blender 乱码名字
        with open(obj_out_path, 'r') as f: lines = f.readlines()
        with open(obj_out_path, 'w') as f:
            clean_name = obj_filename.replace('.obj', '')
            for line in lines:
                if line.startswith('o ') or line.startswith('g '):
                    f.write(f"o {clean_name}\n")
                else: f.write(line)

        print(f"[{count+1}] 保留并随机缩放: {obj_filename} (比例: L/W={length/width:.1f}, W/H={width/height:.1f})")
        count += 1

    except Exception as e:
        pass

print("="*60)
print("自动清洗汇报：")
print(f"  - 成功导出(大概率是刀): {count} 个模型")
print(f"  - 因几何特征不符被扔掉: {dropped_count} 个杂物")
print(f"新模型保存在: {OUTPUT_DIR}")
print("="*60)