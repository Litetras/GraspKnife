import os

# ==========================================
# 开启全局代理（对应你的 v2rayA 端口配置）
# ==========================================
os.environ["http_proxy"] = "http://127.0.0.1:20171"
os.environ["https_proxy"] = "http://127.0.0.1:20171"
os.environ["all_proxy"] = "socks5://127.0.0.1:20170"

import json
import objaverse
import trimesh

def download_and_convert(json_file, output_dir):
    # 1. 读取 UID 列表
    with open(json_file, 'r', encoding='utf-8') as f:
        uids = json.load(f)

    # 核心修改：设置多进程数量
    WORKERS = 1 
    
    print(f"准备开启 {WORKERS} 个进程下载 {len(uids)} 个模型，起飞...")
    os.makedirs(output_dir, exist_ok=True)

    # 2. 批量多进程下载
    objects = objaverse.load_objects(
        uids=uids, 
        download_processes=WORKERS  # <--- 修改这里：从 1 改成 WORKERS(8)
    )

    print(f"下载完成，开始提取并转换为 .obj 格式到 {output_dir} ...")

    # 3. 提取并转换格式
    success_count = 0
    for uid, glb_path in objects.items():
        obj_filename = f"knife_{uid[:6]}.obj"
        obj_path = os.path.join(output_dir, obj_filename)

        if os.path.exists(obj_path):
            continue

        try:
            scene = trimesh.load(glb_path)
            scene.export(obj_path)
            
            success_count += 1
            if success_count % 100 == 0:
                print(f"已转换并保存 {success_count} 个模型...")

        except Exception as e:
            print(f"[跳过] 模型 {uid} 几何损坏或无法转换: {e}")

    print("=" * 50)
    print(f"大功告成！成功提取并转换 {success_count} 个 .obj 文件。")

if __name__ == "__main__":
    JSON_LIST_PATH = "all_knife_no_sword_uids.json"
    TARGET_DIR = "/home/zyp/pan1/knife_4k"
    
    download_and_convert(JSON_LIST_PATH, TARGET_DIR)