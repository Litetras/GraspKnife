import os
import trimesh
import time
import objaverse
import json
import random

# ==========================================
# 代理配置区域 
# ==========================================
PROXY_URL = "http://127.0.0.1:7897"

os.environ['ALL_PROXY'] = PROXY_URL
os.environ['all_proxy'] = PROXY_URL
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
os.environ['http_proxy'] = PROXY_URL
os.environ['https_proxy'] = PROXY_URL

print(f"已设置网络代理为: {PROXY_URL}")

# ==========================================
# 配置区域 (请根据需要修改路径)
# ==========================================
UID_LIST_FILE = "pure_dust_brush_uids.json"  # 刚才筛选好的刷子 UID 列表
OUTPUT_DIR = "/home/zyp/Desktop/brushes"    # 最终 OBJ 存放路径
TARGET_SCALE = 0.3  # 将模型最长边缩放到 0.3米 (30厘米，比较符合手持刷子的尺寸)
PREFIX = "brush" # 导出的文件名前缀
MAX_RETRIES = 5     # 最大重试次数

# ==========================================
# 1. 加载筛选好的 UID 列表
# ==========================================
if not os.path.exists(UID_LIST_FILE):
    raise FileNotFoundError(f"找不到UID文件: {UID_LIST_FILE}，请先运行筛选脚本！")

with open(UID_LIST_FILE, 'r', encoding='utf-8') as f:
    brush_uids = json.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"已加载 {len(brush_uids)} 个手持除尘刷/床刷 UID，输出目录: {OUTPUT_DIR}")

# ==========================================
# 2. 逐个下载与处理逻辑 (完美抗断线)
# ==========================================
success_count = 0
skip_count = 0

print("开始逐个下载并处理模型...")

for i, uid in enumerate(brush_uids):
    # 【断点续传核心】检查是否已经存在处理好的 OBJ 文件
    obj_filename = f"{PREFIX}_{uid[:8]}.obj"
    obj_out_path = os.path.join(OUTPUT_DIR, obj_filename)
    
    if os.path.exists(obj_out_path):
        skip_count += 1
        # 每跳过 10 个打印一次，避免刷屏
        if skip_count % 10 == 0:
            print(f"  -> ⏭️ 已跳过 {skip_count} 个已存在的模型...")
        continue

    # 如果不存在，则开始下载
    download_success = False
    glb_path = None

    for attempt in range(MAX_RETRIES):
        try:
            # 每次严格只向服务器请求 1 个模型
            objects = objaverse.load_objects(uids=[uid], download_processes=1)
            
            if uid in objects:
                glb_path = objects[uid]
                download_success = True
                break
        except Exception as e:
            print(f"  -> ❌ 模型 [{i+1}/{len(brush_uids)}] 下载失败 (尝试 {attempt+1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(3) # 失败后休息3秒
            else:
                print(f"  -> ⚠️ 放弃当前模型 UID: {uid[:8]}")

    # 如果下载成功，开始转换和缩放
    if download_success and glb_path:
        try:
            scene_or_mesh = trimesh.load(glb_path, force='mesh')

            if isinstance(scene_or_mesh, trimesh.Scene):
                if len(scene_or_mesh.geometry) == 0:
                    continue
                mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
            else:
                mesh = scene_or_mesh

            # 居中与缩放
            mesh.apply_translation(-mesh.centroid)
            max_length = mesh.extents.max()
            if max_length > 0:
                scale_factor = TARGET_SCALE / max_length
                mesh.apply_scale(scale_factor)

            # 导出 OBJ
            mesh.export(obj_out_path)

            # 修复内部名称乱码
            with open(obj_out_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            with open(obj_out_path, 'w', encoding='utf-8') as f:
                clean_name = obj_filename.replace('.obj', '')
                for line in lines:
                    if line.startswith('o ') or line.startswith('g '):
                        f.write(f"o {clean_name}\n")
                    else:
                        f.write(line)

            success_count += 1
            print(f"  -> ✅ 成功 [{i+1}/{len(brush_uids)}]: {obj_filename}")
            
            # 【保护代理的核心】成功后随机休息 1 到 2.5 秒，让代理连接池喘口气
            time.sleep(random.uniform(0.1, 0.3))

        except Exception as e:
            print(f"  -> ⚠️ 模型 [{i+1}/{len(brush_uids)}] 转换时报错，已跳过。")
            pass

print(f"\n🎉 处理完毕！新增生成 {success_count} 个，跳过 {skip_count} 个。")