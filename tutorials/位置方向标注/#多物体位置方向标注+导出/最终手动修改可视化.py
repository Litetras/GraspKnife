import open3d as o3d
import os
import json
import numpy as np
import glob

# ===================== 1. 只看这里的名单 =====================
TARGET_MODELS = [
    # 锤子


    "hammer_24", "hammer_22", "hammer_13", "hammer_9", "hammer_2", "hammer_6", "hammer_10", "hammer_23", "hammer_15", "hammer_28", "hammer_1", "hammer_3", "hammer_27", "hammer_18", "hammer_20",
    "brush_1", "brush_9", "brush_13", "brush_8", "brush_1", "brush_12", "brush_3",
    "knife_21", "knife_18", "knife_17", "knife_12", "knife_3", "knife_10", "knife_20",
    "drill_5",
    "mug_4"
    
]

def visualize_specific_targets(base_dir, json_path):
    print("="*60)
    print("🎯 启动专属名单可视化检查器 (已开启抗渐变渲染)")
    print("💡 轴向颜色提示：[红箭头=X轴 (0)] | [绿箭头=Y轴 (1)] | [蓝箭头=Z轴 (2)]")
    print("操作提示：看完当前模型后，按下键盘上的 [Q] 键即可切换到下一个！")
    print("="*60)

    if not os.path.exists(json_path):
        print(f"❌ 找不到 JSON 文件: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    found_count = 0

    for target_name in TARGET_MODELS:
        if target_name not in annotations:
            print(f"⚠️ 在 JSON 中找不到 {target_name} 的标注数据，跳过。")
            continue
            
        data = annotations[target_name]
        category = data.get("category")
        folder_path = os.path.join(base_dir, category)
        
        # 寻找对应的模型文件 (.obj 或 .glb)
        mesh_files = glob.glob(os.path.join(folder_path, f"{target_name}.obj")) + \
                     glob.glob(os.path.join(folder_path, f"{target_name}.glb"))
                     
        if not mesh_files:
            print(f"⚠️ 找不到模型文件: {target_name}，跳过。")
            continue
            
        mesh_path = mesh_files[0]
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        # 细分网格，消除颜色渐变
        mesh = mesh.subdivide_midpoint(number_of_iterations=2)
        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        
        split_axis = data["split_axis"]
        coords = vertices[:, split_axis]
        mode = data.get("mode", "2_points")
        
        # 初始化全灰
        colors = np.full((len(vertices), 3), [0.7, 0.7, 0.7])
        
        # 标红目标区域
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
        
        print(f"\n👀 正在查看: [{category}] {target_name}")
        print(f"   -> 记录的切分轴为: 【{axis_name}】")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"目标检查: {target_name}", width=1024, height=768)
        vis.add_geometry(mesh)
        vis.add_geometry(coord_frame)
        vis.run()
        vis.destroy_window()
        
        found_count += 1

    print(f"\n🎉 专属名单检查完毕！共查看了 {found_count} 个模型。")

if __name__ == "__main__":
    # 替换为你的实际路径
    BASE_DIR = "/home/zyp/Desktop/dataset_obj"  
    OUTPUT_JSON = "all_dataset_boundaries_auto.json"  
    
    visualize_specific_targets(BASE_DIR, OUTPUT_JSON)