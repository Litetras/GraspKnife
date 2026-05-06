import open3d as o3d
import os
import glob
import json
import numpy as np

# 顶会标准：操作者中心坐标系集合 O 固化语义颜色
# 记忆法则: 前后(Z)->蓝系, 上下(Y)->绿系, 左右(X)->红系
SEMANTIC_COLOR_MAP = {
    "Front": ([0.0, 0.5, 1.0], "🔵 亮蓝色"),  # 亮蓝
    "Back":  ([0.0, 0.2, 0.6], "🟦 暗蓝色"),  # 暗蓝
    "Up":    ([0.0, 1.0, 0.0], "🟢 亮绿色"),  # 亮绿
    "Down":  ([0.0, 0.4, 0.0], "🟩 暗绿色"),  # 暗绿
    "Left":  ([1.0, 0.0, 0.0], "🔴 亮红色"),  # 亮红
    "Right": ([0.6, 0.0, 0.0], "🟥 暗红色")   # 暗红
}

# (备用) 万一 JSON 里有手误标错的词，给个默认灰色
DEFAULT_COLOR = ([0.5, 0.5, 0.5], "⚪ 灰色")

def get_rotation_matrix_from_z_to_target(target_dir):
    """计算从默认的 +Z 轴旋转到目标向量方向的旋转矩阵"""
    target_dir = np.array(target_dir) / np.linalg.norm(target_dir)
    z_dir = np.array([0.0, 0.0, 1.0])
    
    if np.allclose(target_dir, z_dir):
        return np.eye(3)
    elif np.allclose(target_dir, -z_dir):
        return o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])
    else:
        axis = np.cross(z_dir, target_dir)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z_dir, target_dir), -1.0, 1.0))
        return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

def visualize_category_directions(base_dir, json_path):
    print("="*70)
    print("🎯 启动【操作者中心坐标系】可视化检查器 (V2.0)")
    print("💡 颜色语义：[蓝色系=前后(Front/Back)] | [绿色系=上下(Up/Down)] | [红色系=左右(Left/Right)]")
    print("操作提示：观察当前模型后，按下键盘上的 [Q] 键即可切换到下一个！")
    print("="*70)

    if not os.path.exists(json_path):
        print(f"❌ 找不到 JSON 文件: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        category_annotations = json.load(f)

    for category_folder, directions in category_annotations.items():
        folder_path = os.path.join(base_dir, category_folder)
        mesh_files = glob.glob(os.path.join(folder_path, '*.obj')) + glob.glob(os.path.join(folder_path, '*.glb'))
        
        if not mesh_files:
            continue
            
        mesh_path = mesh_files[0]
        base_name = os.path.basename(mesh_path)

        # 1. 加载网格
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 调成浅灰色，让深色箭头更明显
        
        # 计算物体尺寸以缩放箭头
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        max_extent = np.max(bbox.get_max_bound() - bbox.get_min_bound())
        
        arrow_scale = max_extent * 0.45  # 箭头稍微加长一点，防止被包在物体里
        
        # 添加一个半透明的原点球体，便于确认箭头的发射点
        center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=arrow_scale * 0.1)
        center_sphere.translate(center)
        center_sphere.paint_uniform_color([0.2, 0.2, 0.2])
        
        geometries_to_draw = [mesh, center_sphere]
        
        print("\n" + "="*50)
        print(f"👀 正在查看类别: 【{category_folder}】")
        print(f"   已绑定 {len(directions)} 个操作者视角向量：")
        
        # 2. 为每个语义方向生成对应固定颜色的箭头
        for semantic_label, data in directions.items():
            # 首字母大写，确保兼容性
            semantic_label = semantic_label.capitalize() 
            vector = np.array(data["vector"])
            axis_str = data["axis_str"]
            
            # 获取专属语义颜色
            color_rgb, color_name = SEMANTIC_COLOR_MAP.get(semantic_label, DEFAULT_COLOR)
            
            # 创建箭头
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=arrow_scale * 0.04, 
                cone_radius=arrow_scale * 0.09,
                cylinder_height=arrow_scale * 0.75,
                cone_height=arrow_scale * 0.25
            )
            
            # 旋转箭头指向目标向量
            R = get_rotation_matrix_from_z_to_target(vector)
            arrow.rotate(R, center=(0, 0, 0))
            
            # 将箭头的起点移动到物体中心
            arrow.translate(center)
            
            # 上色并加入渲染列表
            arrow.paint_uniform_color(color_rgb)
            arrow.compute_vertex_normals() # 为箭头加上光照法线
            geometries_to_draw.append(arrow)
            
            print(f"   -> {color_name} 箭头 代表: [{semantic_label}] (物理轴: {axis_str})")
            
        print("="*50)

        # 3. 可视化 (高度稳定的 draw_geometries 接口)
        o3d.visualization.draw_geometries(
            geometries_to_draw, 
            window_name=f"操作者中心坐标系检查: {category_folder}", 
            width=1024, 
            height=768,
            mesh_show_back_face=True
        )

    print("\n🎉 所有类别的数据集标注可视化检查完毕！逻辑闭环完美。")

if __name__ == "__main__":
    BASE_DIR = "/home/zyp/Desktop/dataset_obj"  
    OUTPUT_JSON = "category_grasp_directions.json"                  
    visualize_category_directions(BASE_DIR, OUTPUT_JSON)