import open3d as o3d
import os
import json
import numpy as np

# ===================== 1. Pan 待修复名单 =====================
# name 必须和 obj/glb 文件名完全一致，不含后缀
#
# split_axis:
#   0 = X轴，红箭头方向
#   1 = Y轴，绿箭头方向
#   2 = Z轴，蓝箭头方向
#
# 你在可视化里看到哪个颜色箭头最顺着平底锅把手，就填哪个轴。
TARGET_MODELS = [
    # 示例：
    # {"name": "pan_12345678", "split_axis": 0},
    # {"name": "pan_abcdef12", "split_axis": 1},
    # {"name": "pan_98765432", "split_axis": 2},

    {"name": "pan_766bd21f", "split_axis": 1},
    {"name": "pan_127e4346", "split_axis": 2},
    {"name": "pan_2d47c429", "split_axis": 2},
    {"name": "pan_a7c6610e", "split_axis": 2},
    {"name": "pan_ffebd77d", "split_axis": 2},
    {"name": "pan_ffebd77d", "split_axis": 2},
    {"name": "pan_afa46d89", "split_axis": 2}


]

# ===================== 2. Pan 核心配置 =====================
CATEGORY_FOLDER = "pans"
MODE = "2_points"


def manual_annotate_single_object(mesh_path, split_axis, mode, category_folder, base_name):
    """【Pan 人工救援】弹窗让人工选点标注"""
    axis_name = ["X", "Y", "Z"][split_axis]

    print("\n" + "=" * 60)
    print(f"🔧 [Pan 精准修复] 正在手动标注: {base_name}")
    print(f"⚠️ 当前切分轴: 【{axis_name} 轴】 | 模式: {mode}")
    print("   1. Shift + 左键: 点击【锅身和把手的分界线】")
    print("   2. Shift + 左键: 点击【把手内部任意一点】")
    print("   选完后按 [Q] 键关闭窗口并保存")
    print("=" * 60)

    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if mesh.is_empty():
        print(f"⚠️ 模型为空，放弃: {base_name}")
        return None

    pcd = mesh.sample_points_uniformly(number_of_points=15000)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name=f"Pan 精准修复 ({axis_name}轴): {base_name}",
        width=1024,
        height=768
    )
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
            "category": category_folder,
            "split_axis": split_axis,
            "mode": mode,
            "boundary_coord": boundary_coord,
            "target_is_positive": target_is_positive
        }

    print(f"⚠️ 未选够点，放弃保存 {base_name}。")
    return None


def fix_pan_annotations(base_dir, json_path):
    """
    只修复 Pan 标注。
    会覆盖 pan_dataset_boundaries_auto.json 里对应模型的旧数据。
    """
    if not os.path.exists(json_path):
        print(f"❌ 找不到 JSON 文件: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    fixed_count = 0
    not_found_count = 0

    print("=" * 60)
    print("🚀 启动 Pan 精准修复模式")
    print(f"📁 模型目录: {os.path.join(base_dir, CATEGORY_FOLDER)}")
    print(f"📝 标注文件: {json_path}")
    print("=" * 60)

    folder_path = os.path.join(base_dir, CATEGORY_FOLDER)

    for item in TARGET_MODELS:
        target_name = item["name"]
        split_axis = item["split_axis"]

        mesh_path_obj = os.path.join(folder_path, f"{target_name}.obj")
        mesh_path_glb = os.path.join(folder_path, f"{target_name}.glb")

        mesh_path = None
        if os.path.exists(mesh_path_obj):
            mesh_path = mesh_path_obj
        elif os.path.exists(mesh_path_glb):
            mesh_path = mesh_path_glb

        if mesh_path is None:
            print(f"❌ 找不到模型文件: {target_name}")
            not_found_count += 1
            continue

        result = manual_annotate_single_object(
            mesh_path=mesh_path,
            split_axis=split_axis,
            mode=MODE,
            category_folder=CATEGORY_FOLDER,
            base_name=target_name
        )

        if result is not None:
            annotations[target_name] = result

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotations, f, indent=4)

            print(f"✅ {target_name} 已成功修复并覆盖原数据！")
            fixed_count += 1

    print("\n" + "=" * 60)
    print(f"🎉 Pan 修复任务结束！共修复并更新了 {fixed_count} 个模型。")
    if not_found_count > 0:
        print(f"⚠️ 有 {not_found_count} 个模型未找到，请检查 TARGET_MODELS 里的名字。")
    print("=" * 60)


if __name__ == "__main__":
    # 你的 pan 模型路径如果是：
    # /home/zyp/Desktop/objaverse_dataset/pans
    # 那 BASE_DIR 就写到 objaverse_dataset 这一层
    BASE_DIR = "/home/zyp/Desktop/objaverse_dataset"

    # 你前面 Pan 自动标注生成的 JSON
    OUTPUT_JSON = "pan_dataset_boundaries_auto.json"

    fix_pan_annotations(BASE_DIR, OUTPUT_JSON)