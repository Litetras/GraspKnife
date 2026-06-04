import open3d as o3d
import os
import glob
import json
import numpy as np

# ============================================================
# Pan 专属操作者中心坐标系可视化检查器
# 会遍历所有 pans 模型逐个显示方向箭头
# ============================================================

SEMANTIC_COLOR_MAP = {
    "Front": ([0.0, 0.5, 1.0], "🔵 亮蓝色"),
    "Back":  ([0.0, 0.2, 0.6], "🟦 暗蓝色"),
    "Up":    ([0.0, 1.0, 0.0], "🟢 亮绿色"),
    "Down":  ([0.0, 0.4, 0.0], "🟩 暗绿色"),
    "Left":  ([1.0, 0.0, 0.0], "🔴 亮红色"),
    "Right": ([0.6, 0.0, 0.0], "🟥 暗红色")
}

DEFAULT_COLOR = ([0.5, 0.5, 0.5], "⚪ 灰色")


def get_rotation_matrix_from_z_to_target(target_dir):
    """
    Open3D 的箭头默认沿 +Z 方向生成。
    这个函数把 +Z 方向旋转到目标 vector 方向。
    """
    target_dir = np.array(target_dir, dtype=np.float64)
    norm = np.linalg.norm(target_dir)

    if norm == 0:
        return np.eye(3)

    target_dir = target_dir / norm
    z_dir = np.array([0.0, 0.0, 1.0])

    if np.allclose(target_dir, z_dir):
        return np.eye(3)

    if np.allclose(target_dir, -z_dir):
        return o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])

    axis = np.cross(z_dir, target_dir)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(z_dir, target_dir), -1.0, 1.0))

    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def visualize_pan_category_directions(base_dir, json_path):
    print("=" * 70)
    print("🎯 启动 Pan / 平底锅 操作者中心坐标系可视化检查器")
    print("💡 颜色语义：[蓝色系=Front/Back] | [绿色系=Up/Down] | [红色系=Left/Right]")
    print("操作提示：观察当前 Pan 模型后，按键盘 [Q] 关闭窗口，会切换到下一个")
    print("=" * 70)

    if not os.path.exists(json_path):
        print(f"❌ 找不到 Pan 方向 JSON 文件: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        category_annotations = json.load(f)

    category_folder = "pans"

    if category_folder not in category_annotations:
        print(f"❌ JSON 中没有找到 '{category_folder}' 的方向配置。")
        print(f"   当前 JSON 里包含的类别: {list(category_annotations.keys())}")
        return

    directions = category_annotations[category_folder]

    folder_path = os.path.join(base_dir, category_folder)

    if not os.path.exists(folder_path):
        print(f"❌ 找不到 Pan 模型文件夹: {folder_path}")
        return

    mesh_files = (
        glob.glob(os.path.join(folder_path, "*.obj")) +
        glob.glob(os.path.join(folder_path, "*.glb"))
    )

    mesh_files = sorted(mesh_files)

    if not mesh_files:
        print(f"⚠️ 在 {folder_path} 中没有找到 obj 或 glb 文件。")
        return

    print(f"\n📦 找到 {len(mesh_files)} 个 Pan 模型，开始逐个可视化。")
    print(f"📁 模型目录: {folder_path}")
    print(f"🧭 方向数量: {len(directions)}")

    for idx, mesh_path in enumerate(mesh_files, start=1):
        base_name = os.path.basename(mesh_path)

        mesh = o3d.io.read_triangle_mesh(mesh_path)

        if mesh.is_empty():
            print(f"❌ 模型为空，跳过: {mesh_path}")
            continue

        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.8, 0.8])

        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        max_extent = np.max(bbox.get_max_bound() - bbox.get_min_bound())

        if max_extent <= 0:
            print(f"⚠️ 模型尺寸异常，跳过: {base_name}")
            continue

        arrow_scale = max_extent * 0.45

        center_sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=arrow_scale * 0.1
        )
        center_sphere.translate(center)
        center_sphere.paint_uniform_color([0.2, 0.2, 0.2])

        geometries_to_draw = [mesh, center_sphere]

        print("\n" + "=" * 60)
        print(f"👀 [{idx}/{len(mesh_files)}] 正在查看 Pan 模型: {base_name}")
        print(f"   已绑定 {len(directions)} 个操作者视角向量：")

        for semantic_label, data in directions.items():
            semantic_label = semantic_label.capitalize()

            vector = np.array(data["vector"], dtype=np.float64)
            axis_str = data.get("axis_str", "unknown")

            color_rgb, color_name = SEMANTIC_COLOR_MAP.get(
                semantic_label,
                DEFAULT_COLOR
            )

            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=arrow_scale * 0.04,
                cone_radius=arrow_scale * 0.09,
                cylinder_height=arrow_scale * 0.75,
                cone_height=arrow_scale * 0.25
            )

            R = get_rotation_matrix_from_z_to_target(vector)
            arrow.rotate(R, center=(0, 0, 0))
            arrow.translate(center)

            arrow.paint_uniform_color(color_rgb)
            arrow.compute_vertex_normals()

            geometries_to_draw.append(arrow)

            print(
                f"   -> {color_name} 箭头 代表: [{semantic_label}] "
                f"| 物理轴: {axis_str} | vector: {vector.tolist()}"
            )

        print("=" * 60)
        print("📌 Pan 方向检查建议：")
        print("   Up    应该指向锅口 / 锅面朝上的方向")
        print("   Down  应该指向锅底方向")
        print("   Front / Back / Left / Right 按你定义的操作者视角检查")
        print("   当前窗口按 [Q] 后进入下一个 Pan")
        print("=" * 60)

        o3d.visualization.draw_geometries(
            geometries_to_draw,
            window_name=f"Pan 操作者中心坐标系检查: {base_name}",
            width=1024,
            height=768,
            mesh_show_back_face=True
        )

    print("\n🎉 所有 Pan 模型方向可视化检查完毕！")


if __name__ == "__main__":
    # Pan 模型目录：
    # /home/zyp/Desktop/objaverse_dataset/pans
    # 所以 BASE_DIR 写到 objaverse_dataset 这一层
    BASE_DIR = "/home/zyp/Desktop/objaverse_dataset"

    # Pan 专属方向 JSON
    OUTPUT_JSON = "pan_category_grasp_directions.json"

    visualize_pan_category_directions(BASE_DIR, OUTPUT_JSON)