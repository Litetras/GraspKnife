import open3d as o3d
import json
import os
import glob
import numpy as np

# =====================================================================
# 🎨 语义色彩映射表
# [蓝色系=前后(Front/Back)] | [绿色系=上下(Up/Down)] | [红色系=左右(Left/Right)]
# =====================================================================
SEMANTIC_COLOR_MAP = {
    "Front": [0.0, 0.5, 1.0],
    "Back":  [0.0, 0.2, 0.6],
    "Up":    [0.0, 1.0, 0.0],
    "Down":  [0.0, 0.4, 0.0],
    "Left":  [1.0, 0.0, 0.0],
    "Right": [0.6, 0.0, 0.0]
}

DEFAULT_COLOR = [0.8, 0.8, 0.0]


def create_gripper_lineset(transform_matrix, color, base_length=0.09188, y_width=0.04):
    """使用 Open3D LineSet 创建一个线框夹爪，支持自定义颜色"""
    z_base = -base_length
    z_bite = -base_length * 0.4

    points = [
        [0, 0, z_base],
        [0, 0, z_bite],
        [0, y_width, z_bite],
        [0, -y_width, z_bite],
        [0, y_width, 0],
        [0, -y_width, 0]
    ]

    lines = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5]]
    colors = [color for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    line_set.transform(transform_matrix)
    return line_set


def get_object_key(data):
    """
    用 object.file 作为“同一个物体”的判断依据。
    例如 mug_1.obj -> mug_1
    """
    obj_filename = data.get("object", {}).get("file", "")
    return os.path.splitext(os.path.basename(obj_filename))[0]


def load_json_records(json_dir):
    """
    预加载所有 JSON，并提取 category / object_key，方便快速跳过。
    """
    json_files = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
    json_files = sorted(json_files)

    records = []

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            obj_filename = data.get("object", {}).get("file")
            if not obj_filename:
                print(f"⚠️ {json_path} 中没有找到 object.file 字段，跳过。")
                continue

            category_dir_name = os.path.basename(os.path.dirname(json_path))
            object_key = get_object_key(data)

            records.append({
                "json_path": json_path,
                "data": data,
                "category": category_dir_name,
                "object_key": object_key,
                "obj_filename": obj_filename
            })

        except Exception as e:
            print(f"⚠️ 读取 JSON 失败，跳过: {json_path} | 原因: {e}")

    return records


def find_obj_path(dataset_obj_dir, category_dir_name, obj_filename):
    """
    查找模型路径。
    默认结构：
        dataset_obj/category/xxx.obj

    兜底：
        dataset_obj/**/xxx.obj
    """
    candidate = os.path.join(dataset_obj_dir, category_dir_name, obj_filename)

    if os.path.exists(candidate):
        return candidate

    recursive_matches = glob.glob(
        os.path.join(dataset_obj_dir, "**", obj_filename),
        recursive=True
    )

    if recursive_matches:
        return recursive_matches[0]

    return None


def show_scene_with_key_callbacks(geometries, window_title):
    """
    Open3D 窗口内快捷键控制：

    Q   -> next，下一个 JSON
    S   -> skip_object，跳过当前物体剩余 JSON
    C   -> skip_category，跳过当前类别剩余 JSON
    Esc -> quit，退出整个可视化
    """
    action_holder = {"action": "next"}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=window_title,
        width=1024,
        height=768
    )

    for g in geometries:
        vis.add_geometry(g)

    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True

    def next_callback(vis):
        action_holder["action"] = "next"
        vis.close()
        return False

    def skip_object_callback(vis):
        action_holder["action"] = "skip_object"
        vis.close()
        return False

    def skip_category_callback(vis):
        action_holder["action"] = "skip_category"
        vis.close()
        return False

    def quit_callback(vis):
        action_holder["action"] = "quit"
        vis.close()
        return False

    # Open3D 里一般用大写 ASCII 注册，实际按 q/s/c 通常也能触发
    vis.register_key_callback(ord("Q"), next_callback)
    vis.register_key_callback(ord("S"), skip_object_callback)
    vis.register_key_callback(ord("C"), skip_category_callback)

    # Esc 键 GLFW key code 通常是 256
    vis.register_key_callback(256, quit_callback)

    print("窗口快捷键：Q=下一个 | S=跳过当前物体 | C=跳过当前类别 | Esc=退出")

    vis.run()
    vis.destroy_window()

    return action_holder["action"]


def batch_visualize_grasps_with_window_hotkeys(dataset_obj_dir, json_dir):
    print("=" * 70)
    print("👀 启动【任务导向】语义抓取批量可视化 + 窗口快捷键版")
    print("💡 色彩法则: [蓝色系=Front/Back] | [绿色系=Up/Down] | [红色系=Left/Right]")
    print("窗口快捷键：")
    print("   Q   ：看下一个 JSON")
    print("   S   ：跳过当前物体的剩余 JSON")
    print("   C   ：跳过当前类别的剩余 JSON")
    print("   Esc ：退出整个可视化")
    print("=" * 70)

    records = load_json_records(json_dir)

    if not records:
        print(f"❌ 在 {json_dir} 中找不到有效 JSON 文件！")
        return

    print(f"📦 共找到 {len(records)} 个有效抓取配置文件，准备逐一展示...\n")

    skipped_objects = set()
    skipped_categories = set()

    shown_count = 0
    skipped_count = 0
    missing_obj_count = 0

    for idx, record in enumerate(records, start=1):
        json_path = record["json_path"]
        data = record["data"]
        category_dir_name = record["category"]
        object_key = record["object_key"]
        obj_filename = record["obj_filename"]

        if object_key in skipped_objects:
            skipped_count += 1
            continue

        if category_dir_name in skipped_categories:
            skipped_count += 1
            continue

        obj_path = find_obj_path(
            dataset_obj_dir=dataset_obj_dir,
            category_dir_name=category_dir_name,
            obj_filename=obj_filename
        )

        if obj_path is None:
            print(f"❌ 找不到对应的模型文件: {obj_filename}，跳过。")
            print(f"   JSON: {json_path}")
            missing_obj_count += 1
            continue

        print("\n" + "-" * 70)
        print(f"📺 [{idx}/{len(records)}] 正在展示:")
        print(f"   类别: {category_dir_name}")
        print(f"   物体: {object_key}")
        print(f"   JSON: {os.path.basename(json_path)}")

        task_semantics = data.get("task_semantics", {})
        task_name = task_semantics.get("task", "Unknown Task")
        region = task_semantics.get("region", "Unknown Region")
        orientation = task_semantics.get("orientation", "Unknown")

        gripper_color = SEMANTIC_COLOR_MAP.get(
            orientation.capitalize(),
            DEFAULT_COLOR
        )

        print(f"   🎯 任务意图 : {task_name}")
        print(f"   🧩 接触部位 : {region}")
        print(f"   🧭 夹爪朝向 : {orientation} | 渲染颜色: RGB {gripper_color}")

        mesh = o3d.io.read_triangle_mesh(obj_path)

        if mesh.is_empty():
            print(f"⚠️ 模型为空，跳过: {obj_path}")
            continue

        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])

        scale = data.get("object", {}).get("scale", 1.0)
        if scale != 1.0:
            mesh.scale(scale, center=(0, 0, 0))

        bbox = mesh.get_axis_aligned_bounding_box()
        max_extent = np.max(bbox.get_max_bound() - bbox.get_min_bound())

        coord_size = 0.1 if max_extent <= 0 else max_extent * 0.4
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=coord_size,
            origin=[0, 0, 0]
        )

        transforms = data.get("grasps", {}).get("transforms", [])
        print(f"   ✅ 正在渲染 {len(transforms)} 个夹爪...")

        geometries = [mesh, coord_frame]

        for T_list in transforms:
            T_matrix = np.array(T_list)
            gripper = create_gripper_lineset(
                T_matrix,
                color=gripper_color
            )
            geometries.append(gripper)

        window_title = (
            f"{os.path.basename(json_path)} | "
            f"{task_name} | {region} | {orientation}"
        )

        action = show_scene_with_key_callbacks(
            geometries=geometries,
            window_title=window_title
        )

        shown_count += 1

        if action == "skip_object":
            skipped_objects.add(object_key)
            print(f"⏭️ 已跳过当前物体剩余 JSON: {object_key}")

        elif action == "skip_category":
            skipped_categories.add(category_dir_name)
            print(f"⏭️ 已跳过当前类别剩余 JSON: {category_dir_name}")

        elif action == "quit":
            print("🛑 用户主动退出可视化。")
            break

        else:
            # action == "next"
            pass

    print("\n" + "=" * 70)
    print("🎉 可视化流程结束！")
    print(f"✅ 实际展示: {shown_count} 个 JSON")
    print(f"⏭️ 快速跳过: {skipped_count} 个 JSON")
    print(f"❌ 缺失模型: {missing_obj_count} 个 JSON")
    print("=" * 70)


if __name__ == "__main__":
    DATASET_OBJ_DIR = "/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集/dataset_obj"
    JSON_OUTPUT_DIR = "/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集/task_oriented_grasps_json"

    batch_visualize_grasps_with_window_hotkeys(
        DATASET_OBJ_DIR,
        JSON_OUTPUT_DIR
    )