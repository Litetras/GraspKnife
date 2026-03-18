import trimesh
import numpy as np
import json
import os
from glob import glob

# ================= 配置路径 =================
# 你的 OBJ 模型存放路径
INPUT_OBJ_DIR = "/home/zyp/Desktop/zyp_dataset2/tutorial/tutorial_object_dataset"
# 生成的残缺点云 JSON 存放路径
OUTPUT_JSON_DIR = "/home/zyp/Desktop/zyp_dataset2/tutorial/tutorial_object_dataset/pc_jsons"
# ==========================================

os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

def generate_partial_pc(mesh, num_points=2048):
    """模拟深度相机，生成单视角下的残缺点云"""
    # 1. 在整个表面均匀采样
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points * 3)
    normals = mesh.face_normals[face_indices]

    # 2. 模拟相机视角 (假设相机位于坐标系斜上方 [1.0, 1.0, 1.0] 的位置)
    camera_pos = np.array([1.0, 1.0, 1.0])
    view_rays = camera_pos - points
    # 归一化视线向量
    view_rays = view_rays / np.linalg.norm(view_rays, axis=1, keepdims=True)

    # 3. 剔除背面点 (法线与视线夹角 > 90度，即点积 < 0 的点)
    dots = np.sum(normals * view_rays, axis=1)
    front_facing_mask = dots > 0
    partial_points = points[front_facing_mask]

    # 4. 采样到固定点数 (补齐或下采样至 2048 个点)
    if len(partial_points) == 0: 
        partial_points = points  # 降级保护
    
    if len(partial_points) > num_points:
        indices = np.random.choice(len(partial_points), num_points, replace=False)
    else:
        indices = np.random.choice(len(partial_points), num_points, replace=True)
    partial_points = partial_points[indices]

    # 5. 添加高斯噪声，模拟真实深度相机的误差
    noise = np.random.normal(0, 0.001, partial_points.shape)
    partial_points += noise

    return partial_points

def main():
    obj_files = glob(os.path.join(INPUT_OBJ_DIR, "*.obj"))
    print(f"🔍 找到 {len(obj_files)} 个 OBJ 模型，准备渲染残缺点云...")

    for obj_path in obj_files:
        base_name = os.path.basename(obj_path).replace(".obj", "")
        json_path = os.path.join(OUTPUT_JSON_DIR, f"{base_name}.json")

        # 加载网格
        try:
            mesh = trimesh.load(obj_path, force='mesh')
        except Exception as e:
            print(f"❌ 无法加载 {obj_path}: {e}")
            continue

        # 生成残缺点云
        pc = generate_partial_pc(mesh, num_points=2048)

        # 生成全灰色点云颜色，用于可视化 [150, 150, 150]
        pc_color = np.ones_like(pc) * 150.0

        # 💡 极其重要的一步：构造一个“假”的基准抓取
        # 因为你的 demo_object_pc.py 代码中会强制读取并计算真实抓取的置信度颜色
        # 如果不放一个假的进去，代码执行到 grasp_conf.min() 时会崩溃
        dummy_grasp = np.eye(4).tolist()
        dummy_conf = [1.0]

        # 封装成 GraspGen 兼容的 JSON 格式
        data = {
            "pc": pc.tolist(),
            "pc_color": pc_color.tolist(),
            "grasp_poses": [dummy_grasp],
            "grasp_conf": dummy_conf
        }

        with open(json_path, "w") as f:
            json.dump(data, f)

        print(f"✅ 成功生成点云: {json_path}")

    print("\n🎉 所有残缺点云 JSON 渲染完成！你可以开始推理了。")

if __name__ == "__main__":
    main()