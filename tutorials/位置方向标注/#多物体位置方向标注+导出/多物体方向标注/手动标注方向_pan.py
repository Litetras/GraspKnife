import open3d as o3d
import os
import glob
import json
import numpy as np

# ===================== 轴向映射表 =====================
AXIS_MAP = {
    '+x': [1.0, 0.0, 0.0],
    '-x': [-1.0, 0.0, 0.0],
    '+y': [0.0, 1.0, 0.0],
    '-y': [0.0, -1.0, 0.0],
    '+z': [0.0, 0.0, 1.0],
    '-z': [0.0, 0.0, -1.0]
}

OPPOSITE_AXIS = {
    '+x': '-x',
    '-x': '+x',
    '+y': '-y',
    '-y': '+y',
    '+z': '-z',
    '-z': '+z'
}

VALID_ORIENTATIONS = ['Up', 'Down', 'Left', 'Right', 'Front', 'Back']

OPPOSITE_ORI = {
    'Up': 'Down',
    'Down': 'Up',
    'Front': 'Back',
    'Back': 'Front',
    'Left': 'Right',
    'Right': 'Left'
}


def annotate_pan_directions(base_dir, output_json):
    print("=" * 70)
    print("🧭 启动 Pan / 平底锅 专属操作者中心坐标系标注器")
    print("📚 标注集合 O = {Up, Down, Left, Right, Front, Back}")
    print("💡 轴向颜色提示：[红=X轴] | [绿=Y轴] | [蓝=Z轴]")
    print("🚀 标注 Up 会自动补全 Down，标注 Front 会自动补全 Back")
    print("=" * 70)

    category_folder = "pans"
    folder_path = os.path.join(base_dir, category_folder)

    if not os.path.exists(folder_path):
        print(f"❌ 找不到 Pan 文件夹: {folder_path}")
        return

    mesh_files = (
        glob.glob(os.path.join(folder_path, "*.obj")) +
        glob.glob(os.path.join(folder_path, "*.glb"))
    )

    if not mesh_files:
        print(f"⚠️ 在 {folder_path} 中没有找到 obj 或 glb 文件。")
        return

    # 读取历史方向标注
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            category_annotations = json.load(f)
        print(f"📦 已加载历史方向数据: {output_json}")
    else:
        category_annotations = {}

    # 如果已经标过 pans，可以选择重新覆盖
    if category_folder in category_annotations:
        print(f"⚠️ 检测到已有 Pan 方向标注: {category_folder}")
        choice = input("是否重新标注并覆盖？输入 y 覆盖，其他键退出: ").strip().lower()
        if choice != "y":
            print("已取消重新标注。")
            return

    # 默认取第一个模型作为代表模型
    representative_mesh_path = mesh_files[0]
    base_name = os.path.basename(representative_mesh_path)

    print("\n" + "=" * 70)
    print(f"🏷️ 正在标注类别: 【{category_folder}】")
    print(f"📌 代表模型: {base_name}")
    print("👀 请在弹出的窗口中观察坐标轴。")
    print("   红色箭头 = X 轴")
    print("   绿色箭头 = Y 轴")
    print("   蓝色箭头 = Z 轴")
    print("观察完毕后，按 [Q] 关闭窗口，然后在终端输入方向。")
    print("=" * 70)

    mesh = o3d.io.read_triangle_mesh(representative_mesh_path)

    if mesh.is_empty():
        print(f"❌ 代表模型为空: {representative_mesh_path}")
        return

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    max_extent = np.max(bbox.get_max_bound() - bbox.get_min_bound())

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max_extent * 0.6,
        origin=center
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"观察 Pan 方向: {category_folder}",
        width=1024,
        height=768
    )
    vis.add_geometry(mesh)
    vis.add_geometry(coord_frame)
    vis.run()
    vis.destroy_window()

    directions_for_pan = {}

    print("\n📌 Pan 推荐理解：")
    print("   Up    ：锅口 / 锅面朝上的方向")
    print("   Down  ：锅底方向")
    print("   Front ：你希望定义为操作者前方的方向")
    print("   Left  ：操作者左侧")
    print("   Right ：操作者右侧")
    print("   Back  ：操作者后方")
    print("你只需要输入其中几个方向，反方向会自动补全。")

    while True:
        missing_oris = [
            ori for ori in VALID_ORIENTATIONS
            if ori not in directions_for_pan
        ]

        if not missing_oris:
            print("🎉 Pan 的 6 个标准方向已全部标注完毕！")
            break

        print("\n" + "-" * 50)
        print(f"待标注方向: {missing_oris}")
        print("目前已记录:", list(directions_for_pan.keys()))

        semantic_label = input(
            "💬 请选择一个要标注的语义方向，如 Up / Front / Left，输入 q 结束: "
        ).strip().capitalize()

        if semantic_label == "Q":
            if len(directions_for_pan) == 0:
                print("⚠️ 没有输入任何方向，Pan 方向标注未保存。")
                return
            break

        if semantic_label not in VALID_ORIENTATIONS:
            print(f"❌ 错误！必须输入: {VALID_ORIENTATIONS}")
            continue

        if semantic_label in directions_for_pan:
            print(f"⚠️ 方向 {semantic_label} 已经标注过。")
            continue

        axis_input = input(
            f"👉 哪个物理轴对应 Pan 的 '{semantic_label}' 方向？"
            "请输入 +x, -x, +y, -y, +z, -z: "
        ).strip().lower()

        if axis_input not in AXIS_MAP:
            print("❌ 无效轴向！请输入 +x, -x, +y, -y, +z, 或 -z。")
            continue

        # 记录当前方向
        directions_for_pan[semantic_label] = {
            "axis_str": axis_input,
            "vector": AXIS_MAP[axis_input]
        }

        print(f"✅ 已记录: {semantic_label} -> {axis_input}")

        # 自动记录反方向
        opp_ori = OPPOSITE_ORI[semantic_label]
        opp_axis = OPPOSITE_AXIS[axis_input]

        if opp_ori not in directions_for_pan:
            directions_for_pan[opp_ori] = {
                "axis_str": opp_axis,
                "vector": AXIS_MAP[opp_axis]
            }
            print(
                f"🤖 自动推导: {opp_ori} -> {opp_axis}"
            )

    # 保存
    category_annotations[category_folder] = directions_for_pan

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(category_annotations, f, indent=4)

    print("\n" + "=" * 70)
    print("🏆 Pan 方向标注完成！")
    print(f"📁 已保存至: {output_json}")
    print("=" * 70)


if __name__ == "__main__":
    # 如果你的 pan 模型目录是：
    # /home/zyp/Desktop/objaverse_dataset/pans
    # 那么 BASE_DIR 写到 objaverse_dataset 这一层
    BASE_DIR = "/home/zyp/Desktop/objaverse_dataset"

    OUTPUT_JSON = "pan_category_grasp_directions.json"

    annotate_pan_directions(BASE_DIR, OUTPUT_JSON)