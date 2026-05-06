import open3d as o3d
import os
import glob
import json
import numpy as np

# ===== 核心配置：新增了 'mode' 字段 =====
# mode: "2_points" 适用一刀切（刀、杯子）；"3_points" 适用两刀切截取中间段（电钻）
CATEGORY_CONFIG = {
    "1_knives":       {"split_axis": 0, "mode": "2_points"},
    "2_hammers":      {"split_axis": 2, "mode": "2_points"},
    "3_brushs":       {"split_axis": 2, "mode": "2_points"},
    "4_drills":       {"split_axis": 1, "mode": "3_points"},  # 电钻用 Y 轴，3点模式
    "5_spoons":       {"split_axis": 0, "mode": "2_points"},
    "6_screwdrivers": {"split_axis": 0, "mode": "2_points"},
    "7_mugs":         {"split_axis": 0, "mode": "2_points"}
}

def annotate_dataset(base_dir, output_json):
    print(f"🚀 开始多类别标注任务，数据集目录: {base_dir}")
    
    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"📦 已加载 {len(annotations)} 个历史标注。")
    else:
        annotations = {}

    for category_folder, config in CATEGORY_CONFIG.items():
        split_axis = config["split_axis"]
        mode = config["mode"]
        axis_name = ['X', 'Y', 'Z'][split_axis]
        
        folder_path = os.path.join(base_dir, category_folder)
        if not os.path.exists(folder_path):
            continue
            
        mesh_files = glob.glob(os.path.join(folder_path, '*.obj')) + glob.glob(os.path.join(folder_path, '*.glb'))

        for mesh_path in mesh_files:
            base_name = os.path.basename(mesh_path).split('.')[0]
            if base_name in annotations:
                continue

            print("\n" + "="*60)
            print(f"🏷️  正在标注 [{category_folder}] -> {base_name}")
            print(f"⚠️  当前设定的切分轴是: 【{axis_name} 轴】 | 模式: {mode}")
            
            # --- 动态打印提示语 ---
            if mode == "2_points":
                print("👉  操作指南 (2点一刀切模式):")
                print("   1. Shift + 左键: 点击【部件的分界线】(如刀刃和刀柄交界处)")
                print("   2. Shift + 左键: 点击【你想抓取的目标部位】(⚠️必须点在手柄/目标区域上！)")
            elif mode == "3_points":
                print("👉  操作指南 (3点两刀切模式 - 电钻专属):")
                print("   1. Shift + 左键: 点击【手柄的起点】(如靠近顶部电机的一端)")
                print("   2. Shift + 左键: 点击【手柄的中间】(确认你要抓取的主体)")
                print("   3. Shift + 左键: 点击【手柄的终点】(如靠近底部电池的一端)")
            print("   最后按 [Q] 键关闭窗口并保存")
            print("="*60)
            
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if not mesh.has_vertices():
                continue
                
            pcd = mesh.sample_points_uniformly(number_of_points=15000)
            pcd.paint_uniform_color([0.6, 0.6, 0.6])

            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name=f"标注 ({axis_name}轴): {base_name}", width=1024, height=768)
            vis.add_geometry(pcd) 
            vis.run()
            vis.destroy_window()

            picked = vis.get_picked_points()
            
            # --- 根据不同模式处理点击逻辑 ---
            if mode == "2_points" and len(picked) >= 2:
                points = np.asarray(pcd.points) 
                boundary_coord = float(points[picked[0]][split_axis])
                target_coord = float(points[picked[1]][split_axis])
                target_is_positive = target_coord > boundary_coord

                annotations[base_name] = {
                    "category": category_folder,
                    "split_axis": split_axis,
                    "mode": mode,
                    "boundary_coord": boundary_coord,
                    "target_is_positive": target_is_positive
                }
                save_and_print(output_json, annotations, f"✅ 保存成功! 边界={boundary_coord:.4f}, 目标部位在{axis_name}轴【{'正方向' if target_is_positive else '负方向'}】")

            elif mode == "3_points" and len(picked) >= 3:
                points = np.asarray(pcd.points)
                coord_start = float(points[picked[0]][split_axis])
                coord_mid = float(points[picked[1]][split_axis])
                coord_end = float(points[picked[2]][split_axis])
                
                # 起点和终点构成了切分这个手柄的最小和最大边界
                boundary_min = min(coord_start, coord_end)
                boundary_max = max(coord_start, coord_end)

                annotations[base_name] = {
                    "category": category_folder,
                    "split_axis": split_axis,
                    "mode": mode,
                    "boundary_min": boundary_min,
                    "boundary_max": boundary_max,
                    "target_mid_coord": coord_mid # 保留中间点坐标作为参考
                }
                save_and_print(output_json, annotations, f"✅ 保存成功! 提取区间 [{boundary_min:.4f}, {boundary_max:.4f}]")
                
            else:
                expected_points = 2 if mode == "2_points" else 3
                print(f"⚠️ [警告] 你点击了 {len(picked)} 个点，但当前模式需要 {expected_points} 个点！本次未保存。")

def save_and_print(json_path, data, msg):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(msg)

if __name__ == "__main__":
    BASE_DIR = "/home/zyp/Desktop/dataset_obj"  
    OUTPUT_JSON = "all_dataset_boundaries.json"                  
    annotate_dataset(BASE_DIR, OUTPUT_JSON)