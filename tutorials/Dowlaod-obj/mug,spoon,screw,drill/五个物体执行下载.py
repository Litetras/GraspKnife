import os
import trimesh
import time
import objaverse
import json
import random

# ==========================================
# 代理配置区域
# ==========================================
# 注意：已经去掉了末尾的冒号，确保格式正确！
PROXY_URL = "http://127.0.0.1:7897"

os.environ['ALL_PROXY'] = PROXY_URL
os.environ['all_proxy'] = PROXY_URL
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
os.environ['http_proxy'] = PROXY_URL
os.environ['https_proxy'] = PROXY_URL

print(f"已设置网络代理为: {PROXY_URL}")

# ==========================================
# 类别与参数配置区域
# ==========================================
BASE_OUTPUT_DIR = "/home/zyp/Desktop/objaverse_dataset"

tasks = {
    #"mug": {"uid_file": "mug_uids.json", "output_dir": os.path.join(BASE_OUTPUT_DIR, "mugs"), "target_scale": 0.10, "prefix": "mug"},
    "spoon": {"uid_file": "spoon_uids.json", "output_dir": os.path.join(BASE_OUTPUT_DIR, "spoons"), "target_scale": 0.18, "prefix": "spoon"},
    #"screwdriver": {"uid_file": "screwdriver_uids.json", "output_dir": os.path.join(BASE_OUTPUT_DIR, "screwdrivers"), "target_scale": 0.20, "prefix": "screwdriver"},
    #"electric_drill": {"uid_file": "electric_drill_uids.json", "output_dir": os.path.join(BASE_OUTPUT_DIR, "electric_drills"), "target_scale": 0.25, "prefix": "electric_drill"},
    #"sprayer": {"uid_file": "sprayer_uids.json", "output_dir": os.path.join(BASE_OUTPUT_DIR, "sprayers"), "target_scale": 0.25, "prefix": "sprayer"}
}

MAX_RETRIES = 5

# ==========================================
# 主循环：遍历处理每一个类别
# ==========================================
for category, config in tasks.items():
    print("\n" + "="*60)
    print(f"🚀 开始处理类别: {category.upper()}")
    print("="*60)
    
    uid_file = config["uid_file"]
    out_dir = config["output_dir"]
    target_scale = config["target_scale"]
    prefix = config["prefix"]

    if not os.path.exists(uid_file):
        print(f"⚠️ 找不到UID文件: {uid_file}，跳过该类别！")
        continue

    with open(uid_file, 'r', encoding='utf-8') as f:
        uids = json.load(f)
    
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{category}] 已加载 {len(uids)} 个模型 UID，输出目录: {out_dir}")

    # ==========================================
    # 逐个下载与处理逻辑 (完美抗断线)
    # ==========================================
    success_count = 0
    skip_count = 0

    for i, uid in enumerate(uids):
        # 【断点续传核心】检查是否已经存在处理好的 OBJ 文件
        obj_filename = f"{prefix}_{uid[:8]}.obj"
        obj_out_path = os.path.join(out_dir, obj_filename)
        
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
                print(f"  -> ❌ 模型 [{i+1}/{len(uids)}] 下载失败 (尝试 {attempt+1}/{MAX_RETRIES})")
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
                    scale_factor = target_scale / max_length
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
                print(f"  -> ✅ 成功 [{i+1}/{len(uids)}]: {obj_filename}")
                
                # 【保护代理的核心】成功后随机休息 1 到 2.5 秒，让代理连接池喘口气
                time.sleep(random.uniform(0.1, 0.5))

            except Exception as e:
                print(f"  -> ⚠️ 模型 [{i+1}/{len(uids)}] 转换时报错，已跳过。")
                pass

    print(f"[{category}] 处理完毕！新增生成 {success_count} 个，跳过 {skip_count} 个。")

print("\n🎉 所有类别的批量下载与处理已全部完成！")