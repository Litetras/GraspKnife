import open3d as o3d
import os
import glob
import json
import numpy as np

# ===================== 1. 待修复名单 =====================
# 请确保这些名字与你实际的 obj 文件名（不含后缀）完全一致
TARGET_MODELS = [
# hammer24 22 13 9 2 6 10 23 15 28 1 3 27 18 20
# brush 1 9 13 8 1 12 3
# knife 21 18 17 12 3 10 20
# drill 5
# mug 4
    "hammer_24", "hammer_22", "hammer_13", "hammer_9", "hammer_2", "hammer_6", "hammer_10", "hammer_23", "hammer_15", "hammer_28", "hammer_1", "hammer_3", "hammer_27", "hammer_18", "hammer_20",
    "brush_1", "brush_9", "brush_13", "brush_8", "brush_1", "brush_12", "brush_3",
    "knife_21", "knife_18", "knife_17", "knife_12", "knife_3", "knife_10", "knife_20",
    "drill_5",
    "mug_4"



]

# ===================== 2. 核心配置 =====================
CATEGORY_CONFIG = {
    "1_knives":       {"split_axis": 0, "mode": "2_points"},
    "2_hammers":      {"split_axis": 0, "mode": "2_points"}, # 注意：如果有需要，请根据实际情况改回正确的轴
    "3_brushs":       {"split_axis": 0, "mode": "2_points"},
    "4_drills":       {"split_axis": 0, "mode": "3_points"},
    "5_spoons":       {"split_axis": 0, "mode": "2_points"},
    "6_screwdrivers": {"split_axis": 0, "mode": "2_points"},
    "7_mugs":         {"split_axis": 0, "mode": "2_points"}
}

def manual_annotate_single_object(mesh_path, split_axis, mode, category_folder, base_name):
    """【人工救援】弹窗让人工选点标注"""
    axis_name = ['X', 'Y', 'Z'][split_axis]
    print("\n" + "="*60)
    print(f"🔧 [精准修复] 正在手动标注: {base_name}")
    print(f"⚠️ 当前切分轴: 【{axis_name} 轴】 | 模式: {mode}")
    
    if mode == "2_points":
        print("   1. Shift + 左键: 点击【部件的分界线】")
        print("   2. Shift + 左键: 点击【目标抓取部位（手柄内部）】")
    else:
        print("   1. Shift + 左键: 点击【手柄的起点】")
        print("   2. Shift + 左键: 点击【手柄的中间】")
        print("   3. Shift + 左键: 点击【手柄的终点】")
    print("   选完后按 [Q] 键关闭窗口并保存")
    print("="*60)
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=15000)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"精准修复 ({axis_name}轴): {base_name}", width=1024, height=768)
    vis.add_geometry(pcd) 
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    if mode == "2_points" and len(picked) >= 2:
        points = np.asarray(pcd.points) 
        boundary_coord = float(points[picked[0]][split_axis])
        target_coord = float(points[picked[1]][split_axis])
        target_is_positive = target_coord > boundary_coord
        return {
            "category": category_folder, "split_axis": split_axis, "mode": mode,
            "boundary_coord": boundary_coord, "target_is_positive": target_is_positive
        }
    elif mode == "3_points" and len(picked) >= 3:
        points = np.asarray(pcd.points)
        c_start, c_end = float(points[picked[0]][split_axis]), float(points[picked[2]][split_axis])
        return {
            "category": category_folder, "split_axis": split_axis, "mode": mode,
            "boundary_min": min(c_start, c_end), "boundary_max": max(c_start, c_end)
        }
    else:
        print(f"⚠️ 未选够点，放弃保存 {base_name}。")
        return None

def fix_specific_annotations(base_dir, json_path):
    # 读取现有的完整标注文件
    if not os.path.exists(json_path):
        print(f"❌ 找不到 JSON 文件: {json_path}，请检查路径。")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    fixed_count = 0
    not_found_count = 0

    print("🚀 启动精准修复模式...")

    # 遍历你要修复的名单
    for target_name in TARGET_MODELS:
        found_model = False
        
        # 在所有类别文件夹中寻找这个模型
        for category_folder, config in CATEGORY_CONFIG.items():
            folder_path = os.path.join(base_dir, category_folder)
            if not os.path.exists(folder_path): continue
            
            # 尝试找 obj 或 glb
            mesh_path_obj = os.path.join(folder_path, f"{target_name}.obj")
            mesh_path_glb = os.path.join(folder_path, f"{target_name}.glb")
            
            mesh_path = None
            if os.path.exists(mesh_path_obj): mesh_path = mesh_path_obj
            elif os.path.exists(mesh_path_glb): mesh_path = mesh_path_glb
            
            if mesh_path:
                found_model = True
                split_axis = config["split_axis"]
                mode = config["mode"]
                
                # 弹窗让你手动修复
                result = manual_annotate_single_object(mesh_path, split_axis, mode, category_folder, target_name)
                
                if result is not None:
                    # 覆写旧数据
                    annotations[target_name] = result
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(annotations, f, indent=4)
                    print(f"✅ {target_name} 已成功修复并覆盖原数据！")
                    fixed_count += 1
                break # 找到了就跳出类别循环
                
        if not found_model:
            print(f"❌ 警告：在所有文件夹中都没有找到模型文件 -> {target_name}")
            not_found_count += 1

    print("\n" + "="*60)
    print(f"🎉 修复任务结束！共修复并更新了 {fixed_count} 个模型。")
    if not_found_count > 0:
        print(f"⚠️ 有 {not_found_count} 个模型未能找到文件，请检查 TARGET_MODELS 里的名字是否拼写正确。")
    print("="*60)

if __name__ == "__main__":
    BASE_DIR = "/home/zyp/Desktop/dataset_obj"  
    OUTPUT_JSON = "all_dataset_boundaries_auto.json" # 这里直接指向你原来的总文件                  
    
    fix_specific_annotations(BASE_DIR, OUTPUT_JSON)