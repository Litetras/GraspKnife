import open3d as o3d
import os
import json
import numpy as np
import glob


def visualize_pan_annotations(base_dir, json_path):
    print("=" * 60)
    print("🎨 启动 Pan / 平底锅 标注结果可视化检查器")
    print("🔴 红色区域 = Handle / 把手区域")
    print("⚪ 灰色区域 = Pan body / 锅身区域")
    print("💡 轴向颜色提示：[红箭头=X轴(0)] | [绿箭头=Y轴(1)] | [蓝箭头=Z轴(2)]")
    print("操作提示：看完当前模型后，按键盘 [Q] 切换到下一个")
    print("=" * 60)

    if not os.path.exists(json_path):
        print(f"❌ 找不到 JSON 文件: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if not annotations:
        print("⚠️ JSON 里没有任何标注数据。")
        return

    for base_name, data in annotations.items():
        category = data.get("category", "pans")
        folder_path = os.path.join(base_dir, category)

        mesh_files = (
            glob.glob(os.path.join(folder_path, f"{base_name}.obj")) +
            glob.glob(os.path.join(folder_path, f"{base_name}.glb"))
        )

        if not mesh_files:
            print(f"⚠️ 找不到模型文件，跳过: {base_name}")
            print(f"   搜索路径: {folder_path}")
            continue

        mesh_path = mesh_files[0]
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        if mesh.is_empty():
            print(f"⚠️ 模型为空，跳过: {base_name}")
            continue

        # ================= 抗渐变渲染 =================
        # 细分网格，让颜色边界更清楚
        # 如果模型太大导致很卡，可以把 2 改成 1 或 0
        mesh = mesh.subdivide_midpoint(number_of_iterations=2)
        # ============================================

        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)

        split_axis = data["split_axis"]
        coords = vertices[:, split_axis]
        mode = data.get("mode", "2_points")

        # 默认灰色
        colors = np.full((len(vertices), 3), [0.7, 0.7, 0.7])

        # Pan 默认使用 2_points：
        # 红色 = 把手区域 Handle
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

        else:
            print(f"⚠️ 未知 mode: {mode}，跳过: {base_name}")
            continue

        colors[target_mask] = [1.0, 0.2, 0.2]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        # 坐标轴
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.08,
            origin=center
        )

        axis_names = ["X(红)", "Y(绿)", "Z(蓝)"]
        axis_name = axis_names[split_axis]

        print(f"\n👀 正在查看 Pan: {base_name}")
        print(f"   -> 模型路径: {mesh_path}")
        print(f"   -> 当前切分轴: 【{axis_name}】")
        print(f"   -> 红色区域应该是【平底锅把手】")
        print(f"   -> 灰色区域应该是【锅身】")

        if mode == "2_points":
            print(f"   -> boundary_coord: {data['boundary_coord']:.6f}")
            print(
                f"   -> target_is_positive: {data['target_is_positive']} "
                f"({'正方向为把手' if data['target_is_positive'] else '负方向为把手'})"
            )

        print("   -> 如果红色区域不是把手，说明 split_axis 或 target_is_positive 需要改。")

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Pan 标注检查: {base_name}",
            width=1024,
            height=768
        )
        vis.add_geometry(mesh)
        vis.add_geometry(coord_frame)
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    # 你的 pan 模型目录应该是：
    # /home/zyp/Desktop/objaverse_dataset/pans
    # 所以 BASE_DIR 写到 objaverse_dataset 这一层
    BASE_DIR = "/home/zyp/Desktop/objaverse_dataset"

    # Pan 专属边界标注 JSON
    OUTPUT_JSON = "pan_dataset_boundaries_auto.json"

    visualize_pan_annotations(BASE_DIR, OUTPUT_JSON)