import open3d as o3d
import os
import glob
import json
import numpy as np
import torch
import sys
from PIL import Image
from scipy.ndimage import zoom

# ===================== SAM3 加载 =====================
sys.path.append(r'/home/zyp/GraspGen')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

print(">>> 正在加载 SAM3 分割模型 (这可能需要一点时间)...")
sam3_model = build_sam3_image_model(checkpoint_path="/home/zyp/sam3/zypmodel/sam3/sam3.pt")
sam3_processor = Sam3Processor(sam3_model)
print(">>> SAM3 加载完成！\n")

# ===================== 核心配置 =====================
CATEGORY_CONFIG = {
    "1_knives":       {"split_axis": 0, "mode": "2_points", "prompt": "handle"},
    "2_hammers":      {"split_axis": 0, "mode": "2_points", "prompt": "handle"},
    "3_brushs":       {"split_axis": 0, "mode": "2_points", "prompt": "handle"},
    "4_drills":       {"split_axis": 0, "mode": "3_points", "prompt": "handle"},
    "5_spoons":       {"split_axis": 0, "mode": "2_points", "prompt": "handle"},
    "6_screwdrivers": {"split_axis": 0, "mode": "2_points", "prompt": "handle"},
    "7_mugs":         {"split_axis": 0, "mode": "2_points", "prompt": "handle"}
}

def auto_extract_handle_3d_points(mesh, split_axis, prompt):
    """【自动化组件】调用 SAM3 提取 3D 把手点云"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=1024, height=1024) 
    vis.add_geometry(mesh)
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.85) 
    vis.poll_events()
    vis.update_renderer()

    rgb = np.asarray(vis.capture_screen_float_buffer(False))
    rgb_data = (rgb * 255).astype(np.uint8)
    depth_data = np.asarray(vis.capture_depth_float_buffer(False))
    
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    intrinsic = cam_params.intrinsic.intrinsic_matrix
    extrinsic = cam_params.extrinsic
    vis.destroy_window()

    rgb_image = Image.fromarray(rgb_data)
    inference_state = sam3_processor.set_image(rgb_image)
    output_obj = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)
    
    masks = output_obj["masks"].cpu().numpy()
    scores = output_obj["scores"].cpu().numpy()

    if len(masks) == 0:
        return None

    best_mask = masks[np.argmax(scores)]
    if len(best_mask.shape) == 3:
        best_mask = best_mask[0]
        
    if best_mask.shape != rgb_data.shape[:2]:
        best_mask = zoom(best_mask, (rgb_data.shape[0]/best_mask.shape[0], rgb_data.shape[1]/best_mask.shape[1]), order=0) > 0.5
    final_mask = (best_mask > 0.5)

    valid_pixels = np.where(final_mask & (depth_data > 0))
    ys, xs = valid_pixels[0], valid_pixels[1]
    if len(xs) == 0:
        return None
        
    zs = depth_data[ys, xs]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    X_cam = (xs - cx) * zs / fx
    Y_cam = (ys - cy) * zs / fy
    Z_cam = zs
    
    points_cam = np.vstack((X_cam, Y_cam, Z_cam, np.ones_like(X_cam)))
    cam_to_world = np.linalg.inv(extrinsic)
    points_world = (cam_to_world @ points_cam)[:3, :].T
    
    return points_world


def manual_annotate_single_object(mesh_path, split_axis, mode, category_folder, base_name):
    """【人工救援组件】弹窗让人工选点标注"""
    axis_name = ['X', 'Y', 'Z'][split_axis]
    print("\n" + "="*60)
    print(f"🆘 [人工补漏] 正在手动标注: {base_name}")
    print(f"⚠️ 当前切分轴: 【{axis_name} 轴】 | 模式: {mode}")
    
    if mode == "2_points":
        print("   1. Shift + 左键: 点击【部件的分界线】")
        print("   2. Shift + 左键: 点击【目标抓取部位】")
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
    vis.create_window(window_name=f"手动标注 ({axis_name}轴): {base_name}", width=1024, height=768)
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

def process_dataset(base_dir, output_json):
    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    else:
        annotations = {}

    fallback_queue = [] # 记录需要人工救援的模型列表

    # ================= 阶段一：SAM3 全自动处理 =================
    for category_folder, config in CATEGORY_CONFIG.items():
        split_axis, mode, prompt = config["split_axis"], config["mode"], config["prompt"]
        folder_path = os.path.join(base_dir, category_folder)
        if not os.path.exists(folder_path): continue
            
        mesh_files = glob.glob(os.path.join(folder_path, '*.obj')) + glob.glob(os.path.join(folder_path, '*.glb'))

        for mesh_path in mesh_files:
            base_name = os.path.basename(mesh_path).split('.')[0]
            if base_name in annotations: continue # 已经标过了，直接跳过

            print(f"\n🤖 正在自动处理: [{category_folder}] -> {base_name}")
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh.compute_vertex_normals()
            
            handle_points_3d = auto_extract_handle_3d_points(mesh, split_axis, prompt)
            
            # 如果自动识别失败，加入人工补漏队列
            if handle_points_3d is None:
                print(f"❌ SAM3 未能识别 {base_name}，已加入手动补漏队列。")
                fallback_queue.append({
                    "mesh_path": mesh_path, "category": category_folder, 
                    "split_axis": split_axis, "mode": mode, "base_name": base_name
                })
                continue
                
            # 计算边界
            handle_coords = handle_points_3d[:, split_axis]
            handle_min, handle_max, handle_center = np.min(handle_coords), np.max(handle_coords), np.mean(handle_coords)
            
            obj_bbox = mesh.get_axis_aligned_bounding_box()
            obj_center = (obj_bbox.get_min_bound()[split_axis] + obj_bbox.get_max_bound()[split_axis]) / 2

            if mode == "2_points":
                target_is_positive = handle_center > obj_center
                boundary_coord = handle_min if target_is_positive else handle_max
                annotations[base_name] = {
                    "category": category_folder, "split_axis": split_axis, "mode": mode,
                    "boundary_coord": float(boundary_coord), "target_is_positive": bool(target_is_positive)
                }
                print(f"✅ 自动计算完成! 边界={boundary_coord:.4f}, 方向={'正' if target_is_positive else '负'}")

            elif mode == "3_points":
                annotations[base_name] = {
                    "category": category_folder, "split_axis": split_axis, "mode": mode,
                    "boundary_min": float(handle_min), "boundary_max": float(handle_max)
                }
                print(f"✅ 自动提取区间完成! [{handle_min:.4f}, {handle_max:.4f}]")
                
            # 实时保存，防止奔溃
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=4)

    # ================= 阶段二：人工补漏处理 =================
    if len(fallback_queue) > 0:
        print("\n" + "!"*60)
        print(f"🚨 自动流程结束。共有 {len(fallback_queue)} 个模型需要你手动补漏！")
        print("!"*60)
        
        for item in fallback_queue:
            result = manual_annotate_single_object(
                item["mesh_path"], item["split_axis"], item["mode"], 
                item["category"], item["base_name"]
            )
            if result is not None:
                annotations[item["base_name"]] = result
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(annotations, f, indent=4)
                print(f"✅ 手动保存成功: {item['base_name']}")
    else:
        print("\n🎉 太完美了！所有模型都已自动标注成功，无需人工介入。")

if __name__ == "__main__":
    BASE_DIR = "/home/zyp/Desktop/dataset_obj"  
    OUTPUT_JSON = "all_dataset_boundaries_auto.json"                  
    process_dataset(BASE_DIR, OUTPUT_JSON)
    print("\n🎉 整个数据集的混合标注流水线运行完毕！")