import open3d as o3d
import os
import json
import numpy as np

# ===================== 1. Key/Fork 待修复名单 =====================
# name 必须和 obj/glb 文件名完全一致，不含后缀
#
# split_axis:
#   0 = X轴，红箭头方向
#   1 = Y轴，绿箭头方向
#   2 = Z轴，蓝箭头方向
#
# 你在可视化里看到哪个颜色箭头最顺着目标区域，就填哪个轴。
TARGET_MODELS = [
    # 示例：
    # {"name": "fork_3", "split_axis": 0, "target_region": "Handle"},
    # {"name": "key_8", "split_axis": 1, "target_region": "Head"},
    {"name": "fork_6", "split_axis": 0, "target_region": "Handle"},
]

# ===================== 2. Key/Fork 核心配置 =====================
CATEGORY_FOLDER = "9_forks"
MODE = "2_points"


def manual_annotate_single_object(mesh_path, split_axis, mode, category_folder, base_name, target_region):
    """【Key/Fork 人工修复】弹窗让人工选点标注"""
    axis_name = ["X", "Y", "Z"][split_axis]

    print("\n" + "=" * 60)
    print(f"🔧 [Key/Fork 精准修复] 正在手动标注: {base_name}")
    print(f"⚠️ 当前切分轴: 【{axis_name} 轴】 | 模式: {mode}")
    print(f"   1. Shift + 左键: 点击【主体和 {target_region} 的分界线】")
    print(f"   2. Shift + 左键: 点击【{target_region} 内部任意一点】")
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
        window_name=f"Key/Fork 精准修复 ({axis_name}轴): {base_name}",
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
            "target_region": target_region,
            "boundary_coord": boundary_coord,
            "target_is_positive": target_is_positive
        }

    print(f"⚠️ 未选够点，放弃保存 {base_name}。")
    return None


def fix_key_fork_annotations(base_dir, json_path):
    """
    只修复 Key/Fork 标注。
    会覆盖 key_fork_dataset_boundaries_auto.json 里对应模型的旧数据。
    """
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
    else:
        print(f"⚠️ 找不到 JSON 文件，将新建: {json_path}")
        annotations = {}

    fixed_count = 0
    not_found_count = 0

    print("=" * 60)
    print("🚀 启动 Key/Fork 精准修复模式")
    print(f"📁 模型目录: {os.path.join(base_dir, CATEGORY_FOLDER)}")
    print(f"📝 标注文件: {json_path}")
    print("=" * 60)

    folder_path = os.path.join(base_dir, CATEGORY_FOLDER)

    for item in TARGET_MODELS:
        target_name = item["name"]
        split_axis = item["split_axis"]
        target_region = item.get("target_region", "Handle")

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
            base_name=target_name,
            target_region=target_region
        )

        if result is not None:
            annotations[target_name] = result

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotations, f, indent=4)

            print(f"✅ {target_name} 已成功修复并覆盖原数据！")
            fixed_count += 1

    print("\n" + "=" * 60)
    print(f"🎉 Key/Fork 修复任务结束！共修复并更新了 {fixed_count} 个模型。")
    if not_found_count > 0:
        print(f"⚠️ 有 {not_found_count} 个模型未找到，请检查 TARGET_MODELS 里的名字。")
    print("=" * 60)


if __name__ == "__main__":
    # 你的 key/fork 模型路径：
    # /home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集/dataset_obj/9_forks
    # /home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集/dataset_obj/10_keys
    # 所以 BASE_DIR 写到 dataset_obj 这一层
    BASE_DIR = "/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集/dataset_obj"

    # 你前面 Key/Fork 标注生成的 JSON
    OUTPUT_JSON = "key_fork_dataset_boundaries_auto.json"

    fix_key_fork_annotations(BASE_DIR, OUTPUT_JSON)
