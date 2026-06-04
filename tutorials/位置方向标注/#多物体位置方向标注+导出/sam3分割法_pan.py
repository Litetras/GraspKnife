import glob
import json
import os
import re
import shutil
import time

import numpy as np
import open3d as o3d

# ===================== Spatula 全手动位置标注 =====================
# 运行后先在终端统一指定一次 split_axis：
#   0 = X 轴，红色
#   1 = Y 轴，绿色
#   2 = Z 轴，蓝色
#
# 之后每个模型都用同一个轴手动点两下：
#   1. 点击主体和 Handle 的分界线
#   2. 点击 Handle 所在一侧
#   关闭窗口后自动写入 JSON
# ==================================================================

CATEGORY_FOLDER = "11_spatulas"
TARGET_REGION_NAME = "Handle"
MODE = "2_points"

BASE_DIR = "/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集/dataset_obj"
OUTPUT_JSON = "spatula_dataset_boundaries_auto.json"

OVERWRITE_EXISTING = True
BACKUP_EXISTING_JSON = True
POINT_SAMPLE_COUNT = 20000


def natural_key(text):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", text)
    ]


def ask_split_axis_once():
    print("=" * 70)
    print("Spatula 全手动位置标注")
    print("请统一指定本批 spatula 的切分轴：")
    print("  0 = X 轴，红色")
    print("  1 = Y 轴，绿色")
    print("  2 = Z 轴，蓝色")
    print("建议：选择最顺着【把手 <-> 铲头】长方向的轴。")
    print("=" * 70)

    while True:
        raw = input("请输入 split_axis (0/1/2): ").strip()
        if raw in {"0", "1", "2"}:
            return int(raw)
        print("输入无效，请输入 0、1 或 2。")


def load_annotations(output_json):
    if not os.path.exists(output_json):
        return {}

    with open(output_json, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if BACKUP_EXISTING_JSON:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{output_json}.manual_backup_{timestamp}"
        shutil.copy2(output_json, backup_path)
        print(f"🛟 已备份旧 JSON: {backup_path}")

    return annotations


def save_annotations(annotations, output_json):
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4)


def make_coord_frame_for_mesh(mesh):
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    max_extent = float(np.max(bbox.get_max_bound() - bbox.get_min_bound()))
    size = max(0.08, max_extent * 0.45)
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=center)


def manual_annotate_single_object(mesh_path, split_axis):
    base_name = os.path.splitext(os.path.basename(mesh_path))[0]
    axis_name = ["X", "Y", "Z"][split_axis]

    print("\n" + "=" * 70)
    print(f"正在手动标注: {base_name}")
    print(f"当前统一切分轴: {axis_name}({split_axis})")
    print(f"目标区域: {TARGET_REGION_NAME}")
    print("操作:")
    print(f"  1. Shift + 左键：点击【主体和 {TARGET_REGION_NAME} 的分界线】")
    print(f"  2. Shift + 左键：点击【{TARGET_REGION_NAME} 所在一侧】")
    print("  3. 按 Q 关闭窗口并保存")
    print("=" * 70)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        print(f"⚠️ 模型为空，跳过: {mesh_path}")
        return None

    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=POINT_SAMPLE_COUNT)
    pcd.paint_uniform_color([0.62, 0.62, 0.62])
    coord_frame = make_coord_frame_for_mesh(mesh)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name=f"Spatula 手动标注 {base_name} | axis={axis_name}",
        width=1200,
        height=850,
    )
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    if len(picked) < 2:
        print(f"⚠️ 未选够 2 个点，放弃保存: {base_name}")
        return None

    points = np.asarray(pcd.points)
    boundary_coord = float(points[picked[0]][split_axis])
    target_coord = float(points[picked[1]][split_axis])
    target_is_positive = target_coord > boundary_coord

    result = {
        "category": CATEGORY_FOLDER,
        "split_axis": split_axis,
        "mode": MODE,
        "target_region": TARGET_REGION_NAME,
        "boundary_coord": boundary_coord,
        "target_is_positive": bool(target_is_positive),
    }

    print(
        f"✅ 保存标注: {base_name} | "
        f"boundary={boundary_coord:.6f} | "
        f"{TARGET_REGION_NAME} 在 {'正方向' if target_is_positive else '负方向'}"
    )
    return result


def process_spatula_manual_dataset(base_dir, output_json):
    folder_path = os.path.join(base_dir, CATEGORY_FOLDER)
    if not os.path.exists(folder_path):
        print(f"❌ 找不到模型文件夹: {folder_path}")
        return

    mesh_files = (
        glob.glob(os.path.join(folder_path, "*.obj"))
        + glob.glob(os.path.join(folder_path, "*.glb"))
    )
    mesh_files = sorted(mesh_files, key=lambda path: natural_key(os.path.basename(path)))

    if not mesh_files:
        print(f"⚠️ 在 {folder_path} 中没有找到 obj 或 glb 文件。")
        return

    split_axis = ask_split_axis_once()
    annotations = load_annotations(output_json)

    print(f"\n📦 找到 {len(mesh_files)} 个 spatula 模型。")
    print(f"📁 模型目录: {folder_path}")
    print(f"📝 输出 JSON: {output_json}")
    print(f"覆盖已有标注: {OVERWRITE_EXISTING}")

    saved_count = 0
    skipped_count = 0

    for mesh_path in mesh_files:
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]

        if base_name in annotations and not OVERWRITE_EXISTING:
            print(f"⏭️ 已存在且不覆盖，跳过: {base_name}")
            skipped_count += 1
            continue

        result = manual_annotate_single_object(mesh_path, split_axis)
        if result is None:
            skipped_count += 1
            continue

        annotations[base_name] = result
        save_annotations(annotations, output_json)
        saved_count += 1

    print("\n" + "=" * 70)
    print("Spatula 全手动位置标注完成")
    print(f"✅ 保存/覆盖: {saved_count} 个")
    print(f"⏭️ 跳过/未保存: {skipped_count} 个")
    print(f"📁 输出: {output_json}")
    print("=" * 70)


if __name__ == "__main__":
    process_spatula_manual_dataset(BASE_DIR, OUTPUT_JSON)
