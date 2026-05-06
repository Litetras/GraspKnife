import open3d as o3d
import os
import glob
import json
import numpy as np

def annotate_hammers(mesh_dir, output_json):
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            annotations = json.load(f)
        print(f"已加载 {len(annotations)} 个历史标注。")
    else:
        annotations = {}

    mesh_files = glob.glob(os.path.join(mesh_dir, '*.obj')) + glob.glob(os.path.join(mesh_dir, '*.glb'))
    
    for mesh_path in mesh_files:
        base_name = os.path.basename(mesh_path).split('.')[0]
        
        if base_name in annotations:
            continue

        print(f"\n[{base_name}] 正在打开...")
        print("操作指南:")
        print("  1. 按住 [Shift] 键 + 鼠标左键点击模型上的点。")
        print("  2. 第一点: 点击【锤柄(handle)与锤头(head)的分界线】")
        print("  3. 第二点: 点击【锤柄(handle)上的任意位置】(用于判断方向)")
        print("  4. 选完两个点后，按 [Q] 键或点击右上角 [X] 关闭窗口进入下一个。")

        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        # --- 生成密集点云以供点击 ---
        # 均匀采样 10000 个点，确保表面布满可选的顶点
        pcd = mesh.sample_points_uniformly(number_of_points=10000)
        pcd.paint_uniform_color([0.7, 0.7, 0.7]) # 涂成灰色方便观察
        # -----------------------------------

        bbox = mesh.get_axis_aligned_bounding_box()
        extents = bbox.get_max_bound() - bbox.get_min_bound()
        major_axis = int(np.argmax(extents))

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=f"标注: {base_name}", width=1024, height=768)
        
        vis.add_geometry(pcd) 
        vis.run()
        vis.destroy_window()

        picked_indices = vis.get_picked_points()

        if len(picked_indices) >= 2:
            points = np.asarray(pcd.points) 
            
            p1 = points[picked_indices[0]]
            boundary_coord = float(p1[major_axis])
            
            p2 = points[picked_indices[1]]
            handle_coord = float(p2[major_axis])
            
            handle_is_positive = handle_coord > boundary_coord

            annotations[base_name] = {
                "major_axis": major_axis,
                "boundary_coord": boundary_coord,
                "handle_is_positive": handle_is_positive
            }
            
            with open(output_json, 'w') as f:
                json.dump(annotations, f, indent=4)
                
            print(f"✅ 保存成功! 边界={boundary_coord:.4f}, 锤柄在正向={handle_is_positive}")
        else:
            print(f"⚠️ [警告] 你点击的点少于 2 个！跳过此模型，下次运行会重新提示。")

    print(f"\n🎉 所有锤子模型标注完成！数据已保存至: {output_json}")

if __name__ == "__main__":
    # 修改为锤子的模型路径
    MESH_DIR = "/home/zyp/Desktop/hammers_cleaned_aligned"  
    # 保存的 JSON 文件名也相应修改
    OUTPUT_JSON = "hammer_boundaries.json"                  
    
    annotate_hammers(MESH_DIR, OUTPUT_JSON)