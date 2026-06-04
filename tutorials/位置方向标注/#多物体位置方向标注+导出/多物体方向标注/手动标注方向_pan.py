import glob
import json
import os

import numpy as np
import open3d as o3d

# ===================== 轴向映射表 =====================
AXIS_MAP = {
    "+x": [1.0, 0.0, 0.0],
    "-x": [-1.0, 0.0, 0.0],
    "+y": [0.0, 1.0, 0.0],
    "-y": [0.0, -1.0, 0.0],
    "+z": [0.0, 0.0, 1.0],
    "-z": [0.0, 0.0, -1.0],
}

OPPOSITE_AXIS = {
    "+x": "-x",
    "-x": "+x",
    "+y": "-y",
    "-y": "+y",
    "+z": "-z",
    "-z": "+z",
}

VALID_ORIENTATIONS = ["Up", "Down", "Left", "Right", "Front", "Back"]

OPPOSITE_ORI = {
    "Up": "Down",
    "Down": "Up",
    "Front": "Back",
    "Back": "Front",
    "Left": "Right",
    "Right": "Left",
}

CATEGORY_CONFIG = {
    "11_spatulas": {
        "display_name": "Spatula",
        "representative": None,  # None 表示自动取第一个模型；也可以写 "spatula_1.obj"
        "tips": [
            "Up    ：铲子正面/可用面朝上的方向",
            "Down  ：铲子背面方向",
            "Front ：铲头指向或你定义的操作前方",
            "Back  ：与 Front 相反",
            "Left  ：操作者左侧",
            "Right ：操作者右侧",
        ],
    }
}


def find_representative_mesh(folder_path, representative):
    if representative:
        path = os.path.join(folder_path, representative)
        return path if os.path.exists(path) else None

    mesh_files = (
        glob.glob(os.path.join(folder_path, "*.obj"))
        + glob.glob(os.path.join(folder_path, "*.glb"))
    )
    return sorted(mesh_files)[0] if mesh_files else None


def show_representative_mesh(mesh_path, category_folder, display_name):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise ValueError(f"代表模型为空: {mesh_path}")

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    max_extent = np.max(bbox.get_max_bound() - bbox.get_min_bound())

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max_extent * 0.6,
        origin=center,
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"观察 {display_name} 方向: {category_folder}",
        width=1024,
        height=768,
    )
    vis.add_geometry(mesh)
    vis.add_geometry(coord_frame)
    vis.run()
    vis.destroy_window()


def annotate_one_category(base_dir, category_folder, config, category_annotations):
    display_name = config["display_name"]
    folder_path = os.path.join(base_dir, category_folder)

    if not os.path.exists(folder_path):
        print(f"❌ 找不到 {display_name} 文件夹: {folder_path}")
        return False

    representative_mesh_path = find_representative_mesh(
        folder_path,
        config.get("representative"),
    )
    if representative_mesh_path is None:
        print(f"⚠️ 在 {folder_path} 中没有找到 obj 或 glb 文件。")
        return False

    if category_folder in category_annotations:
        print(f"⚠️ 检测到已有 {display_name} 方向标注: {category_folder}")
        choice = input("是否重新标注并覆盖？输入 y 覆盖，其他键跳过: ").strip().lower()
        if choice != "y":
            print(f"已跳过 {category_folder}。")
            return False

    print("\n" + "=" * 70)
    print(f"🏷️ 正在标注类别: 【{category_folder}】({display_name})")
    print(f"📌 代表模型: {os.path.basename(representative_mesh_path)}")
    print("👀 请在弹出的窗口中观察坐标轴。")
    print("   红色箭头 = X 轴")
    print("   绿色箭头 = Y 轴")
    print("   蓝色箭头 = Z 轴")
    print("观察完毕后，按 [Q] 关闭窗口，然后在终端输入方向。")
    print("=" * 70)

    show_representative_mesh(representative_mesh_path, category_folder, display_name)

    directions = {}

    print(f"\n📌 {display_name} 推荐理解：")
    for line in config["tips"]:
        print(f"   {line}")
    print("你只需要输入其中几个方向，反方向会自动补全。")

    while True:
        missing_oris = [ori for ori in VALID_ORIENTATIONS if ori not in directions]
        if not missing_oris:
            print(f"🎉 {display_name} 的 6 个标准方向已全部标注完毕！")
            break

        print("\n" + "-" * 50)
        print(f"待标注方向: {missing_oris}")
        print("目前已记录:", list(directions.keys()))

        semantic_label = input(
            "💬 请选择一个要标注的语义方向，如 Up / Front / Left，输入 q 结束: "
        ).strip().capitalize()

        if semantic_label == "Q":
            if len(directions) == 0:
                print(f"⚠️ 没有输入任何方向，{display_name} 方向标注未保存。")
                return False
            break

        if semantic_label not in VALID_ORIENTATIONS:
            print(f"❌ 错误！必须输入: {VALID_ORIENTATIONS}")
            continue

        if semantic_label in directions:
            print(f"⚠️ 方向 {semantic_label} 已经标注过。")
            continue

        axis_input = input(
            f"👉 哪个物理轴对应 {display_name} 的 '{semantic_label}' 方向？"
            "请输入 +x, -x, +y, -y, +z, -z: "
        ).strip().lower()

        if axis_input not in AXIS_MAP:
            print("❌ 无效轴向！请输入 +x, -x, +y, -y, +z, 或 -z。")
            continue

        directions[semantic_label] = {
            "axis_str": axis_input,
            "vector": AXIS_MAP[axis_input],
        }
        print(f"✅ 已记录: {semantic_label} -> {axis_input}")

        opp_ori = OPPOSITE_ORI[semantic_label]
        opp_axis = OPPOSITE_AXIS[axis_input]
        if opp_ori not in directions:
            directions[opp_ori] = {
                "axis_str": opp_axis,
                "vector": AXIS_MAP[opp_axis],
            }
            print(f"🤖 自动推导: {opp_ori} -> {opp_axis}")

    category_annotations[category_folder] = directions
    return True


def annotate_spatula_directions(base_dir, output_json):
    print("=" * 70)
    print("🧭 启动 Spatula 操作者中心坐标系标注器")
    print("📚 标注集合 O = {Up, Down, Left, Right, Front, Back}")
    print("💡 轴向颜色提示：[红=X轴] | [绿=Y轴] | [蓝=Z轴]")
    print("🚀 标注 Up 会自动补全 Down，标注 Front 会自动补全 Back")
    print("=" * 70)

    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            category_annotations = json.load(f)
        print(f"📦 已加载历史方向数据: {output_json}")
    else:
        category_annotations = {}

    saved_count = 0
    for category_folder, config in CATEGORY_CONFIG.items():
        if annotate_one_category(base_dir, category_folder, config, category_annotations):
            saved_count += 1
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(category_annotations, f, indent=4)
            print(f"✅ 已保存 {category_folder} 到: {output_json}")

    print("\n" + "=" * 70)
    print(f"🏆 Spatula 方向标注完成！本次更新类别数: {saved_count}")
    print(f"📁 已保存至: {output_json}")
    print("=" * 70)


if __name__ == "__main__":
    BASE_DIR = "/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集/dataset_obj"
    OUTPUT_JSON = "spatula_category_grasp_directions.json"

    annotate_spatula_directions(BASE_DIR, OUTPUT_JSON)
