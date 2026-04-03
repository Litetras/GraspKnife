import os
import trimesh
import time
import objaverse
import json

# ==========================================
# 代理配置区域 (根据你的 v2rayA 面板设置)
# ==========================================
# 推荐使用 HTTP 端口 20171 (如果想走路由规则可以使用 20172)
PROXY_URL = "http://127.0.0.1:20171"

os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
os.environ['http_proxy'] = PROXY_URL
os.environ['https_proxy'] = PROXY_URL

# 如果你发现 HTTP 代理不好用，可以注释掉上面四行，改用下面的 SOCKS5 代理：
# SOCKS5_URL = "socks5://127.0.0.1:20170"
# os.environ['ALL_PROXY'] = SOCKS5_URL
# os.environ['all_proxy'] = SOCKS5_URL

print(f"已设置网络代理为: {PROXY_URL}")

# ==========================================
# 配置区域 (请根据需要修改路径)
# ==========================================
# 1. 修改为上一个脚本生成的羊角锤 UID 列表文件名
UID_LIST_FILE = "filtered_tool_hammers_uids.json"  

# 2. 修改输出目录
OUTPUT_DIR = "/home/zyp/Desktop/hammers"  

# 3. 修改目标尺寸：标准羊角锤一般长约 30-35cm，这里设为 0.3米
TARGET_SCALE = 0.3  

# ==========================================
# 1. 加载筛选好的 UID 列表
# ==========================================
if not os.path.exists(UID_LIST_FILE):
    raise FileNotFoundError(f"找不到UID文件: {UID_LIST_FILE}，请先运行筛选脚本！")

with open(UID_LIST_FILE, 'r', encoding='utf-8') as f:
    hammer_uids = json.load(f)

print(f"已加载 {len(hammer_uids)} 个羊角锤/工具锤 UID。")

# 可选：如果你只想先测试下载前10个，取消下面这行的注释
# hammer_uids = hammer_uids[:10]

# ==========================================
# 2. 带有重试机制的下载
# ==========================================
print("开始下载模型...")
MAX_RETRIES = 3
objects = {}

for attempt in range(MAX_RETRIES):
    try:
        objects = objaverse.load_objects(
            uids=hammer_uids,
            download_processes=1  # 单进程更稳定，避免网络死锁
        )
        print("下载完成！")
        break
    except Exception as e:
        print(f"下载尝试 {attempt + 1} 失败: {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(3)
        else:
            print("达到最大重试次数，请检查网络。")

if not objects:
    print("没有下载到任何模型，程序退出。")
    exit()

# ==========================================
# 3. 转换、缩放与导出 OBJ
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"开始处理模型，输出目录: {OUTPUT_DIR}")

count = 0
for uid, glb_path in objects.items():
    try:
        # 读取 GLB 文件
        scene_or_mesh = trimesh.load(glb_path, force='mesh')

        # 如果是场景(Scene)，合并为单个网格(Mesh)
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                continue
            mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
        else:
            mesh = scene_or_mesh

        # --- 核心预处理 (非常重要，适合机器人抓取) ---
        # 1. 居中：将模型质心移到坐标原点 (0,0,0)
        mesh.apply_translation(-mesh.centroid)
        
        # 2. 缩放：统一缩放到指定大小 (0.3米)
        max_length = mesh.extents.max()
        if max_length > 0:
            scale_factor = TARGET_SCALE / max_length
            mesh.apply_scale(scale_factor)

        # 3. 修改导出的文件名前缀为 claw_hammer_
        obj_filename = f"claw_hammer_{uid[:8]}.obj"
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

        print(f"[{count+1}/{len(objects)}] 成功处理: {obj_filename}")
        count += 1

    except Exception as e:
        print(f"处理 UID {uid} 时出错: {e}")

print(f"\n全部处理完毕！共生成 {count} 个模型，保存在: {OUTPUT_DIR}")