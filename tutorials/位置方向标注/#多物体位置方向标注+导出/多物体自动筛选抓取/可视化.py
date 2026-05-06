import open3d as o3d
import json
import os
import glob
import numpy as np

# =====================================================================
# 🎨 语义色彩映射表：与之前的可视化保持绝对一致，形成肌肉记忆
# [蓝色系=前后(Front/Back)] | [绿色系=上下(Up/Down)] | [红色系=左右(Left/Right)]
# =====================================================================
SEMANTIC_COLOR_MAP = {
    "Front": [0.0, 0.5, 1.0],  # 亮蓝
    "Back":  [0.0, 0.2, 0.6],  # 暗蓝
    "Up":    [0.0, 1.0, 0.0],  # 亮绿
    "Down":  [0.0, 0.4, 0.0],  # 暗绿
    "Left":  [1.0, 0.0, 0.0],  # 亮红
    "Right": [0.6, 0.0, 0.0]   # 暗红
}

DEFAULT_COLOR = [0.8, 0.8, 0.0] # 默认黄色，防止出错

def create_gripper_lineset(transform_matrix, color, base_length=0.09188, y_width=0.04):
    """使用 Open3D LineSet 创建一个线框夹爪，支持自定义颜色"""
    z_base = -base_length
    z_bite = -base_length * 0.4
    
    # 定义夹爪的 6 个关键点 (TCP在原点 0,0,0)
    points = [
        [0, 0, z_base],         # 0: base point
        [0, 0, z_bite],         # 1: split point
        [0, y_width, z_bite],   # 2: left finger base
        [0, -y_width, z_bite],  # 3: right finger base
        [0, y_width, 0],        # 4: left finger tip (bite point)
        [0, -y_width, 0]        # 5: right finger tip (bite point)
    ]
    
    lines = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5]]
    
    # 给夹爪涂上对应的语义颜色
    colors = [color for _ in range(len(lines))]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # 将变换矩阵应用到夹爪上
    line_set.transform(transform_matrix)
    return line_set

def batch_visualize_grasps(dataset_obj_dir, json_dir):
    print("="*70)
    print("👀 启动【任务导向】语义抓取批量可视化 (V2.0)")
    print("💡 色彩法则: [蓝色系=Front/Back] | [绿色系=Up/Down] | [红色系=Left/Right]")
    print("操作提示：看完当前模型后，关闭窗口 (或按 Q / Esc) 即可自动切换到下一个！")
    print("="*70)

    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)

    if not json_files:
        print(f"❌ 在 {json_dir} 中找不到任何 JSON 文件！")
        return

    print(f"📦 共找到 {len(json_files)} 个抓取配置文件，准备逐一展示...\n")

    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        obj_filename = data.get("object", {}).get("file")
        if not obj_filename:
            print(f"⚠️ {json_path} 中没有找到 object.file 字段，跳过。")
            continue

        category_dir_name = os.path.basename(os.path.dirname(json_path))
        obj_path = os.path.join(dataset_obj_dir, category_dir_name, obj_filename)

        if not os.path.exists(obj_path):
            print(f"❌ 找不到对应的模型文件: {obj_path}，跳过。")
            continue

        print("\n" + "-"*50)
        print(f"📺 正在展示: [{category_dir_name}] -> {os.path.basename(json_path)}")

        # ================= 核心升级点：提取语义信息 =================
        task_semantics = data.get("task_semantics", {})
        task_name = task_semantics.get("task", "Unknown Task")
        region = task_semantics.get("region", "Unknown Region")
        orientation = task_semantics.get("orientation", "Unknown")
        
        # 根据 orientation 获取专属颜色
        gripper_color = SEMANTIC_COLOR_MAP.get(orientation.capitalize(), DEFAULT_COLOR)
        
        print(f"   🎯 任务意图 : {task_name}")
        print(f"   🧩 接触部位 : {region}")
        print(f"   🧭 夹爪朝向 : {orientation} (渲染颜色: RGB {gripper_color})")
        # ============================================================

        # 加载模型
        mesh = o3d.io.read_triangle_mesh(obj_path)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 涂成灰色

        scale = data.get("object", {}).get("scale", 1.0)
        if scale != 1.0:
            mesh.scale(scale, center=(0, 0, 0))

        # 加入一个世界坐标系原点（红X，绿Y，蓝Z），大小根据物体尺寸自适应
        bbox = mesh.get_axis_aligned_bounding_box()
        max_extent = np.max(bbox.get_max_bound() - bbox.get_min_bound())
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max_extent * 0.4, origin=[0, 0, 0])

        transforms = data.get("grasps", {}).get("transforms", [])
        print(f"   ✅ 正在渲染 {len(transforms)} 个夹爪...")

        geometries = [mesh, coord_frame]

        # 生成所有夹爪（传入专属颜色）
        for T_list in transforms:
            T_matrix = np.array(T_list)
            gripper = create_gripper_lineset(T_matrix, color=gripper_color)
            geometries.append(gripper)

        # 渲染当前场景，标题加上详尽的语义信息
        window_title = f"{os.path.basename(json_path)} | {task_name} | {region} | {orientation}"
        o3d.visualization.draw_geometries(
            geometries, 
            window_name=window_title, 
            width=1024, 
            height=768, 
            mesh_show_back_face=True
        )

    print("\n🎉 所有抓取数据已批量展示完毕！")

if __name__ == "__main__":
    # 配置你的两个总目录
    DATASET_OBJ_DIR = "/home/zyp/Desktop/dataset_obj"
    JSON_OUTPUT_DIR = "/home/zyp/Desktop/task_oriented_grasps_json"
    
    batch_visualize_grasps(DATASET_OBJ_DIR, JSON_OUTPUT_DIR)