import argparse
import importlib.util
import json
import os
import shutil
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np


def install_open3d_optional_dependency_stubs():
    """
    sam3_gen 环境里可能没有 plotly/dash。Open3D 导入 visualization 时会顺手导入
    draw_plotly，但本脚本只用桌面窗口，不用 plotly 网页绘图，所以给缺失模块放一个
    最小占位，避免无关依赖导致 Open3D 导入失败。
    """
    if importlib.util.find_spec("plotly") is None:
        plotly_module = types.ModuleType("plotly")
        graph_objects_module = types.ModuleType("plotly.graph_objects")

        class _DummyGraphObject:
            def __init__(self, *args, **kwargs):
                pass

            def show(self, *args, **kwargs):
                raise RuntimeError("plotly is not installed; draw_plotly is unavailable.")

        graph_objects_module.Scatter3d = _DummyGraphObject
        graph_objects_module.Mesh3d = _DummyGraphObject
        graph_objects_module.Figure = _DummyGraphObject
        plotly_module.graph_objects = graph_objects_module
        sys.modules["plotly"] = plotly_module
        sys.modules["plotly.graph_objects"] = graph_objects_module

    dash_module = types.ModuleType("dash")
    dash_module.html = types.SimpleNamespace()
    dash_module.dcc = types.SimpleNamespace()

    class _DummyDash:
        def __init__(self, *args, **kwargs):
            pass

    dash_module.Dash = _DummyDash
    sys.modules["dash"] = dash_module

    if importlib.util.find_spec("yaml") is None:
        yaml_module = types.ModuleType("yaml")

        def _unavailable_yaml(*args, **kwargs):
            raise RuntimeError("PyYAML is not installed; Open3D-ML YAML helpers are unavailable.")

        yaml_module.safe_load = _unavailable_yaml
        yaml_module.safe_dump = _unavailable_yaml
        sys.modules["yaml"] = yaml_module

    # Open3D 顶层导入会尝试加载 open3d.ml；本脚本完全不用 ML 数据集/训练接口，
    # 预先占位可以避免 pandas/pytz/dateutil 等无关依赖缺失导致桌面可视化也无法启动。
    sys.modules.setdefault("open3d.ml", types.ModuleType("open3d.ml"))


install_open3d_optional_dependency_stubs()
import open3d as o3d


DATASET_ROOT = Path("/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集")
DEFAULT_DATASET_OBJ_DIR = DATASET_ROOT / "dataset_obj"
DEFAULT_JSON_DIR = DATASET_ROOT / "task_oriented_grasps_json"

SEMANTIC_COLOR_MAP = {
    "Front": [0.0, 0.5, 1.0],
    "Back": [0.0, 0.2, 0.6],
    "Up": [0.0, 1.0, 0.0],
    "Down": [0.0, 0.4, 0.0],
    "Left": [1.0, 0.0, 0.0],
    "Right": [0.6, 0.0, 0.0],
}

DEFAULT_GRIPPER_COLOR = [0.9, 0.8, 0.0]
BAD_PICK_CENTER_COLOR = [1.0, 0.05, 0.05]
DISPLAY_KEEP_ALL = 0


def create_gripper_lineset(transform_matrix, color, base_length=0.09188, y_width=0.04):
    z_base = -base_length
    z_bite = -base_length * 0.4

    points = [
        [0, 0, z_base],
        [0, 0, z_bite],
        [0, y_width, z_bite],
        [0, -y_width, z_bite],
        [0, y_width, 0],
        [0, -y_width, 0],
    ]
    lines = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5]]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    line_set.transform(transform_matrix)
    return line_set


def sample_segment(p0, p1, samples_per_segment):
    return np.linspace(p0, p1, samples_per_segment, dtype=np.float64)


def create_gripper_sample_points(
    transform_matrix,
    color,
    base_length=0.09188,
    y_width=0.04,
    samples_per_segment=14,
):
    """把线框夹爪采样成点，避免 VisualizerWithEditing 忽略 LineSet。"""
    z_base = -base_length
    z_bite = -base_length * 0.4

    local_points = np.asarray([
        [0, 0, z_base],
        [0, 0, z_bite],
        [0, y_width, z_bite],
        [0, -y_width, z_bite],
        [0, y_width, 0],
        [0, -y_width, 0],
    ], dtype=np.float64)
    lines = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5]]

    homogeneous = np.concatenate(
        [local_points, np.ones((len(local_points), 1), dtype=np.float64)],
        axis=1,
    )
    world_points = (transform_matrix @ homogeneous.T).T[:, :3]

    sampled = []
    for i, j in lines:
        sampled.append(sample_segment(world_points[i], world_points[j], samples_per_segment))
    sampled = np.concatenate(sampled, axis=0)
    colors = np.tile(np.asarray(color, dtype=np.float64), (len(sampled), 1))
    return sampled, colors


def create_center_marker_points(center, radius):
    """红色小十字 + 中心点。点击附近任一点都会映射回最近的抓取中心。"""
    offsets = np.asarray([
        [0.0, 0.0, 0.0],
        [radius, 0.0, 0.0],
        [-radius, 0.0, 0.0],
        [0.0, radius, 0.0],
        [0.0, -radius, 0.0],
        [0.0, 0.0, radius],
        [0.0, 0.0, -radius],
    ], dtype=np.float64)
    return center[None, :] + offsets


def get_object_key(data):
    obj_filename = data.get("object", {}).get("file", "")
    return Path(obj_filename).stem


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_json_records(args):
    if args.json_files:
        json_paths = [Path(path) for path in args.json_files]
    else:
        json_paths = sorted(args.json_dir.glob("**/*.json"))

    records = []
    for json_path in json_paths:
        try:
            data = load_json(json_path)
        except Exception as exc:
            print(f"⚠️ 读取失败，跳过: {json_path} | {exc}")
            continue

        transforms = data.get("grasps", {}).get("transforms")
        obj_file = data.get("object", {}).get("file")
        if not isinstance(transforms, list) or not obj_file:
            # debug report 之类的 JSON 没有 object/grasps，直接跳过。
            continue

        task_semantics = data.get("task_semantics", {}) or {}
        category = json_path.parent.name
        object_key = get_object_key(data)

        if args.categories and category not in args.categories:
            continue
        if args.objects and object_key not in args.objects:
            continue
        if args.regions and task_semantics.get("region") not in args.regions:
            continue
        if args.orientations and task_semantics.get("orientation") not in args.orientations:
            continue
        if args.tasks and task_semantics.get("task") not in args.tasks:
            continue

        records.append(
            {
                "json_path": json_path,
                "data": data,
                "category": category,
                "object_key": object_key,
                "obj_filename": Path(obj_file).name,
                "task_semantics": task_semantics,
                "transform_count": len(transforms),
            }
        )

    return records


def find_obj_path(dataset_obj_dir, category_dir_name, obj_filename):
    direct = dataset_obj_dir / category_dir_name / obj_filename
    if direct.exists():
        return direct

    matches = sorted(dataset_obj_dir.glob(f"**/{obj_filename}"))
    return matches[0] if matches else None


def choose_display_indices(total_count, max_display_grasps):
    if max_display_grasps == DISPLAY_KEEP_ALL or total_count <= max_display_grasps:
        return list(range(total_count))
    if max_display_grasps <= 0:
        return list(range(total_count))
    return np.linspace(0, total_count - 1, max_display_grasps, dtype=int).tolist()


def build_scene(
    record,
    dataset_obj_dir,
    point_size,
    max_display_grasps,
    center_radius_scale,
    object_sample_points,
    gripper_line_samples,
    pick_radius_scale,
):
    data = record["data"]
    obj_path = find_obj_path(dataset_obj_dir, record["category"], record["obj_filename"])
    if obj_path is None:
        raise FileNotFoundError(f"找不到 OBJ: {record['obj_filename']}")

    mesh = o3d.io.read_triangle_mesh(str(obj_path))
    if mesh.is_empty():
        raise ValueError(f"OBJ 为空: {obj_path}")

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.70, 0.70, 0.70])

    scale = float(data.get("object", {}).get("scale", 1.0))
    if scale != 1.0:
        mesh.scale(scale, center=(0, 0, 0))

    bbox = mesh.get_axis_aligned_bounding_box()
    max_extent = float(np.max(bbox.get_max_bound() - bbox.get_min_bound()))
    marker_radius = max(0.003, max_extent * center_radius_scale)
    pick_radius = max(marker_radius * 3.0, max_extent * pick_radius_scale)

    task_semantics = record["task_semantics"]
    orientation = str(task_semantics.get("orientation", "")).capitalize()
    gripper_color = SEMANTIC_COLOR_MAP.get(orientation, DEFAULT_GRIPPER_COLOR)

    transforms = data.get("grasps", {}).get("transforms", [])
    display_indices = choose_display_indices(len(transforms), max_display_grasps)
    centers = []
    valid_display_indices = []
    center_point_grasp_indices = []
    marker_point_grasp_indices = []
    gripper_point_grasp_indices = []
    marker_points = []
    gripper_points = []
    gripper_colors = []

    for original_idx in display_indices:
        transform_list = transforms[original_idx]
        transform = np.asarray(transform_list, dtype=np.float64)
        if transform.shape != (4, 4):
            continue
        valid_display_indices.append(original_idx)
        centers.append(transform[:3, 3])
        center_point_grasp_indices.append(original_idx)

        marker = create_center_marker_points(transform[:3, 3], marker_radius)
        marker_points.append(marker)
        marker_point_grasp_indices.extend([original_idx] * len(marker))

        sampled_points, sampled_colors = create_gripper_sample_points(
            transform,
            color=gripper_color,
            samples_per_segment=gripper_line_samples,
        )
        gripper_points.append(sampled_points)
        gripper_colors.append(sampled_colors)
        gripper_point_grasp_indices.extend([original_idx] * len(sampled_points))

    centers = np.asarray(centers, dtype=np.float64)
    marker_points = (
        np.concatenate(marker_points, axis=0)
        if marker_points else np.empty((0, 3), dtype=np.float64)
    )
    marker_colors = np.tile(
        np.asarray(BAD_PICK_CENTER_COLOR, dtype=np.float64),
        (len(marker_points), 1),
    )

    gripper_points = (
        np.concatenate(gripper_points, axis=0)
        if gripper_points else np.empty((0, 3), dtype=np.float64)
    )
    gripper_colors = (
        np.concatenate(gripper_colors, axis=0)
        if gripper_colors else np.empty((0, 3), dtype=np.float64)
    )

    object_cloud = mesh.sample_points_uniformly(number_of_points=object_sample_points)
    object_points = np.asarray(object_cloud.points)
    object_colors = np.tile(np.asarray([0.66, 0.66, 0.66]), (len(object_points), 1))

    # 所有可视元素合成一个点云，保证 VisualizerWithEditing 一定能显示它们。
    # 同时记录每个点属于哪个原始 grasp index；灰色物体点为 -1。
    all_points = np.concatenate([centers, marker_points, gripper_points, object_points], axis=0)
    center_colors = np.tile(np.asarray(BAD_PICK_CENTER_COLOR), (len(centers), 1))
    all_colors = np.concatenate([center_colors, marker_colors, gripper_colors, object_colors], axis=0)
    point_to_grasp_index = np.asarray(
        center_point_grasp_indices
        + marker_point_grasp_indices
        + gripper_point_grasp_indices
        + [-1] * len(object_points),
        dtype=np.int64,
    )

    editable_cloud = o3d.geometry.PointCloud()
    editable_cloud.points = o3d.utility.Vector3dVector(all_points)
    editable_cloud.colors = o3d.utility.Vector3dVector(all_colors)

    geometries = [editable_cloud]

    return geometries, valid_display_indices, centers, point_to_grasp_index, pick_radius, len(transforms), point_size


def pick_bad_grasps(record, args):
    geometries, display_indices, centers, point_to_grasp_index, pick_radius, total_count, point_size = build_scene(
        record,
        dataset_obj_dir=args.dataset_obj_dir,
        point_size=args.point_size,
        max_display_grasps=args.max_display_grasps,
        center_radius_scale=args.center_radius_scale,
        object_sample_points=args.object_sample_points,
        gripper_line_samples=args.gripper_line_samples,
        pick_radius_scale=args.pick_radius_scale,
    )
    center_count = len(display_indices)

    task = record["task_semantics"].get("task", "Unknown")
    region = record["task_semantics"].get("region", "Unknown")
    orientation = record["task_semantics"].get("orientation", "Unknown")
    title = (
        f"REMOVE BAD | {record['object_key']} | {region}_{orientation} | "
        f"{center_count}/{total_count} shown"
    )

    print("\n" + "-" * 80)
    print(f"JSON : {record['json_path']}")
    print(f"物体 : {record['object_key']}")
    print(f"任务 : {task} | region={region} | orientation={orientation}")
    print(f"数量 : 总计 {total_count} 个，本窗口显示 {center_count} 个")
    if center_count < total_count:
        print("提示 : 当前只显示抽样抓取。要显示全部请加 --max-display-grasps 0")
    print("操作 : Shift + 左键点击任意夹爪彩色点/红色中心 = 标记删除该夹爪")
    print("       Shift + 右键取消选择 | Q 关闭窗口并进入确认")
    print("       关窗后终端可输入 o 跳过当前物体，c 跳过当前类别，q 退出")
    print("颜色 : 灰色=物体采样点 | 绿色/蓝色/红色系=夹爪方向 | 亮红色=可删除中心")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=900)

    for geometry in geometries:
        vis.add_geometry(geometry)

    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.point_size = point_size

    vis.run()
    picked = sorted(set(int(idx) for idx in vis.get_picked_points()))
    vis.destroy_window()

    cloud_points = np.asarray(geometries[0].points)
    valid_picks = []
    invalid_picks = []
    for idx in picked:
        if idx < 0 or idx >= len(cloud_points):
            invalid_picks.append(idx)
            continue

        grasp_idx = int(point_to_grasp_index[idx])
        if grasp_idx >= 0:
            valid_picks.append(grasp_idx)
            continue

        # 兜底：如果用户点到了夹爪附近的灰色物体点，也允许按最近抓取中心删除。
        # 这比只能点红点宽松，但仍用半径限制避免误删远处抓取。
        picked_point = cloud_points[idx]
        distances = np.linalg.norm(centers - picked_point[None, :], axis=1)
        nearest_local_idx = int(np.argmin(distances))
        if distances[nearest_local_idx] <= pick_radius:
            valid_picks.append(display_indices[nearest_local_idx])
        else:
            invalid_picks.append(idx)

    if invalid_picks:
        print(f"⚠️ 忽略非抓取中心点选择: {invalid_picks[:20]}")

    return sorted(set(valid_picks))


def backup_json(json_path, json_root, backup_root):
    try:
        relative_path = json_path.relative_to(json_root)
    except ValueError:
        relative_path = Path(json_path.name)

    backup_path = backup_root / relative_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(json_path, backup_path)
    return backup_path


def append_removal_log(log_path, json_path, removed_indices, old_count, new_count):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    item = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "json_path": str(json_path),
        "removed_indices": removed_indices,
        "old_count": old_count,
        "new_count": new_count,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_vector(vector):
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector
    return vector / norm


def vector_angle_deg(vector_a, vector_b):
    vector_a = normalize_vector(vector_a)
    vector_b = normalize_vector(vector_b)
    dot = float(np.clip(np.dot(vector_a, vector_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def expand_remove_indices_by_nearby(transforms, remove_indices, position_threshold, angle_threshold_deg):
    """
    可选的“近邻批量删除”模式：
    用户点掉一个坏抓取时，附近位置相近且接近方向相近的抓取通常也是同一簇坏候选。
    只在显式传入 --delete-nearby 时启用，避免误删正常的密集候选。
    """
    selected = sorted(set(remove_indices))
    if not selected:
        return []

    matrices = []
    for transform in transforms:
        matrix = np.asarray(transform, dtype=np.float64)
        matrices.append(matrix if matrix.shape == (4, 4) else None)

    selected_matrices = [matrices[idx] for idx in selected if 0 <= idx < len(matrices) and matrices[idx] is not None]
    if not selected_matrices:
        return selected

    expanded = set(selected)
    for idx, matrix in enumerate(matrices):
        if matrix is None:
            continue
        position = matrix[:3, 3]
        approach = matrix[:3, 2]
        for selected_matrix in selected_matrices:
            selected_position = selected_matrix[:3, 3]
            selected_approach = selected_matrix[:3, 2]
            position_distance = float(np.linalg.norm(position - selected_position))
            approach_angle = vector_angle_deg(approach, selected_approach)
            if position_distance <= position_threshold and approach_angle <= angle_threshold_deg:
                expanded.add(idx)
                break

    return sorted(expanded)


def remove_indices_from_json(record, remove_indices, args):
    json_path = record["json_path"]
    data = load_json(json_path)
    transforms = data.get("grasps", {}).get("transforms", [])

    exact_remove_indices = sorted(set(remove_indices))
    if args.delete_nearby:
        remove_indices = expand_remove_indices_by_nearby(
            transforms=transforms,
            remove_indices=exact_remove_indices,
            position_threshold=args.nearby_position_threshold,
            angle_threshold_deg=args.nearby_angle_threshold_deg,
        )
        print(
            "🔎 近邻扩展删除: "
            f"点选 {len(exact_remove_indices)} 个 -> 实际删除 {len(remove_indices)} 个 "
            f"(位置≤{args.nearby_position_threshold:.4f}m, 方向≤{args.nearby_angle_threshold_deg:.1f}°)"
        )

    remove_set = set(remove_indices)
    new_transforms = [
        transform for idx, transform in enumerate(transforms)
        if idx not in remove_set
    ]

    if len(new_transforms) == len(transforms):
        print("没有实际删除。")
        return 0

    if args.dry_run:
        print(f"[dry-run] 将删除 {len(transforms) - len(new_transforms)} 个抓取，不写入文件。")
        return len(transforms) - len(new_transforms)

    backup_path = backup_json(json_path, args.json_dir, args.backup_dir)
    data["grasps"]["transforms"] = new_transforms
    save_json(data, json_path)

    verified_data = load_json(json_path)
    verified_count = len(verified_data.get("grasps", {}).get("transforms", []))
    if verified_count != len(new_transforms):
        print(f"⚠️ 保存校验异常: 期望 {len(new_transforms)}，实际读取 {verified_count}")

    append_removal_log(
        log_path=args.backup_dir / "manual_removed_grasps.jsonl",
        json_path=json_path,
        removed_indices=remove_indices,
        old_count=len(transforms),
        new_count=len(new_transforms),
    )

    print(f"✅ 已删除 {len(transforms) - len(new_transforms)} 个抓取")
    print(f"   数量: {len(transforms)} -> {len(new_transforms)} (校验读取: {verified_count})")
    print(f"   备份: {backup_path}")
    return len(transforms) - len(new_transforms)


def ask_next_action(remove_indices, assume_yes):
    """
    返回值：
    - apply: 应用当前点选删除，然后继续
    - next: 不删除，继续下一个 JSON
    - skip_object: 跳过当前物体后续 JSON
    - skip_category: 跳过当前类别后续 JSON
    - quit: 退出整个流程
    """
    if not remove_indices:
        if assume_yes:
            return "next"
        print("未选中待删除抓取。")
        answer = input(
            "回车/ n=下一个 | o=跳过当前物体 | c=跳过当前类别 | q=退出: "
        ).strip().lower()
        if answer == "o":
            return "skip_object"
        if answer == "c":
            return "skip_category"
        if answer == "q":
            return "quit"
        return "next"

    print(f"选中待删除 index: {remove_indices}")
    if assume_yes:
        return "apply"

    answer = input(
        "y=确认删除 | n/回车=不删下一个 | o=删后跳当前物体 | c=删后跳当前类别 | q=删后退出: "
    ).strip().lower()
    if answer == "y":
        return "apply"
    if answer == "o":
        return "apply_skip_object"
    if answer == "c":
        return "apply_skip_category"
    if answer == "q":
        return "apply_quit"
    return "next"


def main():
    args = parse_args()
    if args.backup_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.backup_dir = args.json_dir / f"_manual_remove_backups_{stamp}"

    records = iter_json_records(args)
    print("=" * 80)
    print("手动点击剔除瑕疵抓取")
    print(f"JSON dir: {args.json_dir}")
    print(f"OBJ dir : {args.dataset_obj_dir}")
    print(f"待检查 JSON 数: {len(records)}")
    print(f"备份目录: {args.backup_dir}")
    if args.json_dir == DATASET_ROOT:
        print("⚠️ 注意: 你当前 --json-dir 指向数据集根目录，会递归扫描多个 task_oriented* 结果目录。")
        print("   如果你只想改旧筛选结果，请指定 --json-dir .../task_oriented_grasps_json")
        print("   如果你只想改 handle_aug 结果，请指定对应的 task_oriented_grasps_json_*_handle_aug 目录。")
    task_dirs = sorted(path for path in args.json_dir.glob("task_oriented*") if path.is_dir())
    if task_dirs:
        print("⚠️ 当前目录下存在多个结果目录，同名 JSON 可能不止一份:")
        for path in task_dirs:
            print(f"   - {path}")
    print("=" * 80)

    if not records:
        print("没有找到可处理的抓取 JSON。")
        return

    total_removed = 0
    processed = 0
    skipped_count = 0
    skipped_objects = set()
    skipped_categories = set()

    for record in records:
        if record["object_key"] in skipped_objects:
            skipped_count += 1
            continue
        if record["category"] in skipped_categories:
            skipped_count += 1
            continue

        try:
            remove_indices = pick_bad_grasps(record, args)
        except Exception as exc:
            print(f"❌ 展示失败，跳过: {record['json_path']} | {exc}")
            continue

        action = ask_next_action(remove_indices, args.yes)
        if action in {"apply", "apply_skip_object", "apply_skip_category", "apply_quit"}:
            total_removed += remove_indices_from_json(record, remove_indices, args)
        else:
            print("跳过删除。")

        processed += 1

        if action in {"skip_object", "apply_skip_object"}:
            skipped_objects.add(record["object_key"])
            print(f"⏭️ 后续跳过当前物体: {record['object_key']}")
        elif action in {"skip_category", "apply_skip_category"}:
            skipped_categories.add(record["category"])
            print(f"⏭️ 后续跳过当前类别: {record['category']}")
        elif action in {"quit", "apply_quit"}:
            print("🛑 用户退出。")
            break

    print("\n" + "=" * 80)
    print(f"完成。检查 JSON: {processed} 个，累计删除抓取: {total_removed} 个")
    print(f"快速跳过 JSON: {skipped_count} 个")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="手动点击剔除任务导向抓取 JSON 中的个别瑕疵抓取。")
    parser.add_argument("--dataset-obj-dir", type=Path, default=DEFAULT_DATASET_OBJ_DIR)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--json-files", nargs="+", default=None, help="只处理指定 JSON 文件。")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--orientations", nargs="+", default=None)
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.add_argument(
        "--center-radius-scale",
        type=float,
        default=0.018,
        help="红色中心小球半径占物体最大包围盒尺寸的比例。",
    )
    parser.add_argument("--pick-radius-scale", type=float, default=0.035)
    parser.add_argument("--object-sample-points", type=int, default=7000)
    parser.add_argument("--gripper-line-samples", type=int, default=18)
    parser.add_argument(
        "--max-display-grasps",
        type=int,
        default=250,
        help="每个窗口最多显示多少个抓取；0 表示显示全部。",
    )
    parser.add_argument("--yes", action="store_true", help="不在终端二次确认，直接删除点选结果。")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--delete-nearby",
        action="store_true",
        help="删除点选抓取附近的一簇相似抓取，适合处理密集重复坏候选。",
    )
    parser.add_argument(
        "--nearby-position-threshold",
        type=float,
        default=0.012,
        help="--delete-nearby 的位置阈值，单位米。",
    )
    parser.add_argument(
        "--nearby-angle-threshold-deg",
        type=float,
        default=10.0,
        help="--delete-nearby 的接近方向夹角阈值。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
