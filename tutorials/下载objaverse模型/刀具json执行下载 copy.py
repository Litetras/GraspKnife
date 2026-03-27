import os
import trimesh
import time
import objaverse
import json
import random  # 【新增】引入随机数库

# ==========================================
# 代理设置
# ==========================================
os.environ["http_proxy"] = "http://127.0.0.1:20171"
os.environ["https_proxy"] = "http://127.0.0.1:20171"

# ==========================================
# 配置区域
# ==========================================
UID_LIST_FILE = "kitchen_knife_pure_uids.json"
OUTPUT_DIR = "/home/zyp/Desktop/knives"

# 【修改】定义你希望模型最终随机分布的尺寸区间 (单位：米)
# 比如：让所有的刀在 15cm(水果刀) 到 40cm(大主厨刀) 之间随机生成
TARGET_MIN_SCALE = 0.15  
TARGET_MAX_SCALE = 0.40  

# ==========================================
# 1. 加载筛选好的 UID 列表
# ==========================================
if not os.path.exists(UID_LIST_FILE):
    raise FileNotFoundError(f"找不到UID文件: {UID_LIST_FILE}，请先运行筛选脚本！")

with open(UID_LIST_FILE, 'r', encoding='utf-8') as f:
    knife_uids = json.load(f)

print(f"已加载 {len(knife_uids)} 个厨房刀具 UID。")

# ==========================================
# 2. 带有重试机制的下载
# ==========================================
# (Objaverse 会自动检查本地缓存，之前下载过的会瞬间跳过，不会重复消耗流量)
print("检查并加载模型缓存...")
MAX_RETRIES = 3
objects = {}

for attempt in range(MAX_RETRIES):
    try:
        objects = objaverse.load_objects(
            uids=knife_uids,
            download_processes=8
        )
        print("模型加载完成！")
        break
    except Exception as e:
        print(f"加载尝试 {attempt + 1} 失败: {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(3)
        else:
            print("达到最大重试次数，请检查网络。")
            exit()

if not objects:
    print("没有加载到任何模型，程序退出。")
    exit()

# ==========================================
# 3. 转换、随机缩放与导出 OBJ
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"开始处理模型，输出目录: {OUTPUT_DIR}")

count = 0
error_count = 0

for uid, glb_path in objects.items():
    try:
        scene_or_mesh = trimesh.load(glb_path, force='mesh')

        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                continue
            mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
        else:
            mesh = scene_or_mesh

        # --- 核心预处理：居中与随机缩放 ---
        
        # 1. 居中：将模型质心移到坐标原点 (0,0,0)
        mesh.apply_translation(-mesh.centroid)
        
        # 2. 获取当前模型包围盒的最大边长
        current_max_length = mesh.extents.max()
        
        # 3. 【核心逻辑】随机生成目标尺寸并缩放
        if current_max_length > 0:
            # 在 0.15m 到 0.40m 之间随机抽取一个长度
            random_target_scale = random.uniform(TARGET_MIN_SCALE, TARGET_MAX_SCALE)
            # 计算缩放比例
            scale_factor = random_target_scale / current_max_length
            # 执行缩放
            mesh.apply_scale(scale_factor)
        else:
            print(f"  -> 跳过 UID {uid[:8]}: 模型没有任何体积 (长度为0)")
            error_count += 1
            continue

        # 导出文件
        obj_filename = f"kitchen_knife_{uid[:8]}.obj"
        obj_out_path = os.path.join(OUTPUT_DIR, obj_filename)
        mesh.export(obj_out_path)

        # 修复 Blender 导入时的名称乱码问题
        with open(obj_out_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(obj_out_path, 'w', encoding='utf-8') as f:
            clean_name = obj_filename.replace('.obj', '')
            for line in lines:
                if line.startswith('o ') or line.startswith('g '):
                    f.write(f"o {clean_name}\n")
                else:
                    f.write(line)

        print(f"[{count+1}] 成功处理: {obj_filename} (已随机缩放至: {random_target_scale:.3f} 米)")
        count += 1

    except Exception as e:
        print(f"处理 UID {uid} 时出错: {e}")
        error_count += 1

print("="*60)
print("处理汇报：")
print(f"  - 成功导出: {count} 把随机尺寸的厨刀")
print(f"  - 异常损坏: {error_count} 个模型")
print(f"最终模型保存在: {OUTPUT_DIR}")
print("="*60)