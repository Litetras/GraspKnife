import open3d as o3d
import os
import json
import numpy as np
import glob

def visualize_annotations(base_dir, json_path):
    print("="*60)
    print("🎨 启动标注结果可视化检查器 (已开启抗渐变渲染)")
    print("💡 轴向颜色提示：[红箭头=X轴 (0)] | [绿箭头=Y轴 (1)] | [蓝箭头=Z轴 (2)]")
    print("操作提示：看完当前模型后，按下键盘上的 [Q] 键即可切换到下一个！")
    print("="*60)

    if not os.path.exists(json_path):
        print(f"❌ 找不到 JSON 文件: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    for base_name, data in annotations.items():
        category = data.get("category")
        folder_path = os.path.join(base_dir, category)
        
        mesh_files = glob.glob(os.path.join(folder_path, f"{base_name}.obj")) + \
                     glob.glob(os.path.join(folder_path, f"{base_name}.glb"))
                     
        if not mesh_files:
            continue
            
        mesh_path = mesh_files[0]
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        # ================= 修复渐变色的核心代码 =================
        # 细分网格 2 次。将大三角形切成无数小三角形，让颜色边界像刀切一样锐利！
        # (如果你的模型本身非常庞大，细分会导致加载变慢，可以把 2 改成 1)
        mesh = mesh.subdivide_midpoint(number_of_iterations=2)
        # ========================================================
        
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        
        split_axis = data["split_axis"]
        coords = vertices[:, split_axis]
        mode = data.get("mode", "2_points")
        
        colors = np.full((len(vertices), 3), [0.7, 0.7, 0.7])
        
        if mode == "2_points":
            boundary_coord = data["boundary_coord"]
            if data["target_is_positive"]:
                target_mask = coords > boundary_coord
            else:
                target_mask = coords < boundary_coord
        elif mode == "3_points":
            b_min = data["boundary_min"]
            b_max = data["boundary_max"]
            target_mask = (coords > b_min) & (coords < b_max)
            
        colors[target_mask] = [1.0, 0.2, 0.2]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=center)
        
        axis_names = ['X(红)', 'Y(绿)', 'Z(蓝)']
        axis_name = axis_names[split_axis]
        
        print(f"\n👀 正在查看: [{category}] {base_name}")
        print(f"   -> 当前切分轴为: 【{axis_name}】")
        print(f"   -> 如果出现'纵向劈开'，请观察哪个颜色的箭头顺着手柄，去 JSON 里把 split_axis 改成对应的数字。")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"可视化: {base_name}", width=1024, height=768)
        vis.add_geometry(mesh)
        vis.add_geometry(coord_frame)
        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    BASE_DIR = "/home/zyp/Desktop/dataset_obj"
    OUTPUT_JSON = "all_dataset_boundaries_auto.json"  
    visualize_annotations(BASE_DIR, OUTPUT_JSON)