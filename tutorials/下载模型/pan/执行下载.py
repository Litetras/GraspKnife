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
# 类别与参数配置区域：只处理 Pan
# ==========================================
BASE_OUTPUT_DIR = "/home/zyp/Desktop/objaverse_dataset"

tasks = {
    "pan": {
        "uid_file": "pan_uids.json",
        "output_dir": os.path.join(BASE_OUTPUT_DIR, "pans"),
        "target_scale": 0.25,
        "prefix": "pan"
    }
}

MAX_RETRIES = 5

# ==========================================
# 主循环：遍历处理 Pan 类别
# ==========================================
for category, config in tasks.items():
    print("\n" + "=" * 60)
    print(f"🚀 开始处理类别: {category.upper()}")
    print("=" * 60)

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

    success_count = 0
    skip_count = 0

    for i, uid in enumerate(uids):
        obj_filename = f"{prefix}_{uid[:8]}.obj"
        obj_out_path = os.path.join(out_dir, obj_filename)

        # 断点续传：已经存在就跳过
        if os.path.exists(obj_out_path):
            skip_count += 1
            if skip_count % 10 == 0:
                print(f"  -> ⏭️ 已跳过 {skip_count} 个已存在的模型...")
            continue

        download_success = False
        glb_path = None

        for attempt in range(MAX_RETRIES):
            try:
                objects = objaverse.load_objects(
                    uids=[uid],
                    download_processes=1
                )

                if uid in objects:
                    glb_path = objects[uid]
                    download_success = True
                    break

            except Exception as e:
                print(
                    f"  -> ❌ 模型 [{i + 1}/{len(uids)}] 下载失败 "
                    f"(尝试 {attempt + 1}/{MAX_RETRIES})"
                )

                if attempt < MAX_RETRIES - 1:
                    time.sleep(3)
                else:
                    print(f"  -> ⚠️ 放弃当前模型 UID: {uid[:8]}")

        if download_success and glb_path:
            try:
                scene_or_mesh = trimesh.load(glb_path, force='mesh')

                if isinstance(scene_or_mesh, trimesh.Scene):
                    if len(scene_or_mesh.geometry) == 0:
                        continue
                    mesh = trimesh.util.concatenate(
                        tuple(scene_or_mesh.geometry.values())
                    )
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

                # 修复 OBJ 内部名称
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
                print(f"  -> ✅ 成功 [{i + 1}/{len(uids)}]: {obj_filename}")

                time.sleep(random.uniform(0.1, 0.5))

            except Exception as e:
                print(f"  -> ⚠️ 模型 [{i + 1}/{len(uids)}] 转换时报错，已跳过。")
                pass

    print(f"[{category}] 处理完毕！新增生成 {success_count} 个，跳过 {skip_count} 个。")

print("\n🎉 Pan 类别的批量下载与处理已全部完成！")