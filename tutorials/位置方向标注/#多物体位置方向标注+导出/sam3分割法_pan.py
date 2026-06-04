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
sam3_model = build_sam3_image_model(
    checkpoint_path="/home/zyp/sam3/zypmodel/sam3/sam3.pt"
)
sam3_processor = Sam3Processor(sam3_model)
print(">>> SAM3 加载完成！\n")

# ===================== Pan 专属配置 =====================
# 只处理平底锅 / 煎锅。
# mode = 2_points:
#   自动或手动找出 handle 与 pan body 的分界线。
#
# split_axis = 0:
#   默认认为平底锅的把手主要沿 X 轴延伸。
#   如果你可视化后发现把手沿 Y 轴或 Z 轴，需要改成 1 或 2。
#
# prompt_candidates:
#   SAM3 会依次尝试这些 prompt，提高识别把手成功率。
CATEGORY_CONFIG = {
    "pans": {
        "split_axis": 0,
        "mode": "2_points",
        "prompt_candidates": [
            "frying pan handle",
            "pan handle",
            "skillet handle",
            "handle"
        ]
    }
}


def auto_extract_handle_3d_points(mesh, split_axis, prompt_candidates):
    """
    【自动化组件】
    调用 SAM3 提取平底锅把手的 3D 点云。
    会依次尝试多个 prompt，只要有一个成功就返回。
    """
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

    for prompt in prompt_candidates:
        print(f"   🔎 SAM3 尝试 prompt: {prompt}")

        try:
            inference_state = sam3_processor.set_image(rgb_image)
            output_obj = sam3_processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

            masks = output_obj["masks"].cpu().numpy()
            scores = output_obj["scores"].cpu().numpy()

            if len(masks) == 0:
                continue

            best_mask = masks[np.argmax(scores)]

            if len(best_mask.shape) == 3:
                best_mask = best_mask[0]

            if best_mask.shape != rgb_data.shape[:2]:
                best_mask = zoom(
                    best_mask,
                    (
                        rgb_data.shape[0] / best_mask.shape[0],
                        rgb_data.shape[1] / best_mask.shape[1]
                    ),
                    order=0
                ) > 0.5

            final_mask = best_mask > 0.5

            valid_pixels = np.where(final_mask & (depth_data > 0))
            ys, xs = valid_pixels[0], valid_pixels[1]

            if len(xs) == 0:
                continue

            zs = depth_data[ys, xs]
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]

            X_cam = (xs - cx) * zs / fx
            Y_cam = (ys - cy) * zs / fy
            Z_cam = zs

            points_cam = np.vstack((X_cam, Y_cam, Z_cam, np.ones_like(X_cam)))
            cam_to_world = np.linalg.inv(extrinsic)
            points_world = (cam_to_world @ points_cam)[:3, :].T

            print(f"   ✅ SAM3 成功识别把手，使用 prompt: {prompt}")
            return points_world

        except Exception as e:
            print(f"   ⚠️ prompt '{prompt}' 识别失败，继续尝试下一个。")
            continue

    return None


def manual_annotate_single_object(mesh_path, split_axis, mode, category_folder, base_name):
    """
    【人工救援组件】
    如果 SAM3 没识别出 pan handle，就弹窗让人工选点标注。
    """
    axis_name = ['X', 'Y', 'Z'][split_axis]

    print("\n" + "=" * 60)
    print(f"🆘 [人工补漏] 正在手动标注 Pan: {base_name}")
    print(f"⚠️ 当前切分轴: 【{axis_name} 轴】 | 模式: {mode}")

    print("   1. Shift + 左键: 点击【锅身和把手的分界线】")
    print("   2. Shift + 左键: 点击【把手所在一侧】")
    print("   选完后按 [Q] 键关闭窗口并保存")
    print("=" * 60)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=15000)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name=f"手动标注 Pan ({axis_name}轴): {base_name}",
        width=1024,
        height=768
    )
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()

    if len(picked) >= 2:
        points = np.asarray(pcd.points)

        boundary_coord = float(points[picked[0]][split_axis])
        target_coord = float(points[picked[1]][split_axis])
        target_is_positive = target_coord > boundary_coord

        return {
            "category": category_folder,
            "split_axis": split_axis,
            "mode": mode,
            "boundary_coord": boundary_coord,
            "target_is_positive": target_is_positive
        }

    print(f"⚠️ 未选够点，放弃保存 {base_name}。")
    return None


def process_pan_dataset(base_dir, output_json):
    """
    只处理 pan 类别。
    输出每个 pan 模型的 handle 边界信息。
    """
    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    else:
        annotations = {}

    fallback_queue = []

    # ================= 阶段一：SAM3 全自动处理 Pan =================
    for category_folder, config in CATEGORY_CONFIG.items():
        split_axis = config["split_axis"]
        mode = config["mode"]
        prompt_candidates = config["prompt_candidates"]

        folder_path = os.path.join(base_dir, category_folder)

        if not os.path.exists(folder_path):
            print(f"❌ 找不到 Pan 模型文件夹: {folder_path}")
            continue

        mesh_files = (
            glob.glob(os.path.join(folder_path, "*.obj")) +
            glob.glob(os.path.join(folder_path, "*.glb"))
        )

        if not mesh_files:
            print(f"⚠️ 在 {folder_path} 中没有找到 obj 或 glb 文件。")
            continue

        print(f"\n📦 找到 {len(mesh_files)} 个 Pan 模型，开始处理...")

        for mesh_path in mesh_files:
            base_name = os.path.basename(mesh_path).split(".")[0]

            if base_name in annotations:
                print(f"⏭️ 已标注过，跳过: {base_name}")
                continue

            print(f"\n🤖 正在自动处理 Pan: {base_name}")

            mesh = o3d.io.read_triangle_mesh(mesh_path)
            mesh.compute_vertex_normals()

            handle_points_3d = auto_extract_handle_3d_points(
                mesh,
                split_axis,
                prompt_candidates
            )

            # 如果自动识别失败，加入人工补漏队列
            if handle_points_3d is None:
                print(f"❌ SAM3 未能识别 {base_name} 的把手，已加入手动补漏队列。")
                fallback_queue.append({
                    "mesh_path": mesh_path,
                    "category": category_folder,
                    "split_axis": split_axis,
                    "mode": mode,
                    "base_name": base_name
                })
                continue

            # 计算 handle 在 split_axis 上的范围
            handle_coords = handle_points_3d[:, split_axis]
            handle_min = np.min(handle_coords)
            handle_max = np.max(handle_coords)
            handle_center = np.mean(handle_coords)

            # 计算整个物体中心
            obj_bbox = mesh.get_axis_aligned_bounding_box()
            obj_center = (
                obj_bbox.get_min_bound()[split_axis] +
                obj_bbox.get_max_bound()[split_axis]
            ) / 2

            # 对 pan 来说：
            # handle_center 在物体中心哪一侧，就认为哪一侧是 Handle。
            target_is_positive = handle_center > obj_center
            boundary_coord = handle_min if target_is_positive else handle_max

            annotations[base_name] = {
                "category": category_folder,
                "split_axis": split_axis,
                "mode": mode,
                "boundary_coord": float(boundary_coord),
                "target_is_positive": bool(target_is_positive)
            }

            print(
                f"✅ 自动计算完成! "
                f"边界={boundary_coord:.4f}, "
                f"把手方向={'正方向' if target_is_positive else '负方向'}"
            )

            # 实时保存，防止崩溃
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, indent=4)

    # ================= 阶段二：人工补漏处理 =================
    if len(fallback_queue) > 0:
        print("\n" + "!" * 60)
        print(f"🚨 自动流程结束。共有 {len(fallback_queue)} 个 Pan 模型需要你手动补漏！")
        print("!" * 60)

        for item in fallback_queue:
            result = manual_annotate_single_object(
                item["mesh_path"],
                item["split_axis"],
                item["mode"],
                item["category"],
                item["base_name"]
            )

            if result is not None:
                annotations[item["base_name"]] = result

                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(annotations, f, indent=4)

                print(f"✅ 手动保存成功: {item['base_name']}")
    else:
        print("\n🎉 所有 Pan 模型都已自动标注成功，无需人工介入。")


if __name__ == "__main__":
    # 如果你的 pan 文件夹是：
    # /home/zyp/Desktop/objaverse_dataset/pans
    # 就保持下面这样。
    BASE_DIR = "/home/zyp/Desktop/objaverse_dataset"

    # 输出 Pan 专属边界标注文件
    OUTPUT_JSON = "pan_dataset_boundaries_auto.json"

    process_pan_dataset(BASE_DIR, OUTPUT_JSON)

    print("\n🎉 Pan 类别的混合标注流水线运行完毕！")