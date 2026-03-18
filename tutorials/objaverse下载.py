import os
import trimesh
import time
import objaverse

# 1. 获取经过人工验证的 LVIS 标注字典
lvis_annotations = objaverse.load_lvis_annotations()

# ==========================================
# 修改点 1：把 'hammer' 改成了 'knife'
# ==========================================
knife_uids = lvis_annotations.get('knife', [])
print(f"一共找到了 {len(knife_uids)} 把刀具模型。")

# 3. 带有网络重试机制的单进程下载
print("开始下载/校验... (采用单进程模式以避免网络波动导致的进程死锁)")

MAX_RETRIES = 3  # 最大重试次数
objects = {}

for attempt in range(MAX_RETRIES):
    try:
        # 修改点 2：传入 knife_uids
        objects = objaverse.load_objects(
            uids=knife_uids,d
            download_processes=1
        )
        print("所有模型下载/校验成功！")
        break  # 成功后跳出重试循环
        
    except Exception as e:
        print(f"第 {attempt + 1} 次下载尝试失败，遇到错误: {e}")
        if attempt < MAX_RETRIES - 1:
            print("等待 3 秒后自动重试...")
            time.sleep(3)
        else:
            print("达到最大重试次数，请检查网络或配置代理后再试。")

# 4. 打印结果验证
if objects:
    print("\n--- 下载结果展示 ---")
    for uid, path in list(objects.items())[:5]:
        print(f"模型 {uid} 本地路径: {path}")


# ========================================================
# 下方是转换与缩放逻辑
# ========================================================

# ==========================================
# 修改点 3：输出文件夹改为 my_knife_objs
# ==========================================
output_obj_dir = "/home/zyp/Desktop/my_knife_objs"
os.makedirs(output_obj_dir, exist_ok=True)

print(f"\n--- 开始将全部 .glb 转换为适合抓取的 .obj ---")

count = 0
# 遍历 objects 字典中的所有模型
for uid, glb_path in objects.items():
    try:
        # 1. 读取 GLB 文件 (force='mesh' 尽量作为单一网格读取)
        scene_or_mesh = trimesh.load(glb_path, force='mesh')

        # 2. 如果读出来是 Scene (包含多个子部件的场景)，将其合并为单个 Mesh
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                print(f"UID {uid} 几何体为空，跳过。")
                continue
            # 合并场景中的所有几何体
            mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
        else:
            mesh = scene_or_mesh

        # ==========================================
        # 核心避坑操作：几何预处理 (极其重要！)
        # ==========================================
        # A. 居中：把物体的质心移动到世界坐标系原点 (0,0,0)
        mesh.apply_translation(-mesh.centroid)
        
        # B. 缩放：获取模型在 XYZ 三个方向上的跨度 (extents)
        # 刀具我们也把它最长的那条边缩放到 0.2 米 (20厘米，常见的水果刀/菜刀尺寸)
        max_length = mesh.extents.max()
        if max_length > 0:
            scale_factor = 0.2 / max_length
            mesh.apply_scale(scale_factor)
        # ==========================================

        # ==========================================
        # 修改点 4：文件名前缀改为 knife_
        # ==========================================
        obj_filename = f"knife_{uid[:6]}.obj"
        obj_out_path = os.path.join(output_obj_dir, obj_filename)
        
        # 3. 导出为 OBJ 文件
        mesh.export(obj_out_path)
        
        # ==========================================
        # 终极修复：暴力修改 OBJ 内部的物体名称标签，导入 Blender 不乱码
        # ==========================================
        with open(obj_out_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        with open(obj_out_path, 'w', encoding='utf-8') as f:
            clean_name = obj_filename.replace('.obj', '')
            for line in lines:
                if line.startswith('o ') or line.startswith('g '):
                    f.write(f"o {clean_name}\n")
                else:
                    f.write(line)
        # ==========================================

        print(f"成功转换、缩放并重命名: {obj_out_path}")
        count += 1
        
    except Exception as e:
        print(f"UID {uid} 转换失败，错误信息: {e}")

print(f"\n全部完成！成功生成 {count} 把极品刀具 OBJ 模型，存放在 {output_obj_dir} 文件夹中。")