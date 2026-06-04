import argparse
import glob
import os
import random

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation


DATASET_ROOT = "/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集"
DEFAULT_GRASPS_DIR = os.path.join(DATASET_ROOT, "grasps")
DEFAULT_OBJ_DIR = os.path.join(DATASET_ROOT, "dataset_obj")

CATEGORY_CONFIG = {
    "fork": {
        "grasp_dir": "frok_grasp",
        "obj_dir": "9_forks",
        "color": [0.0, 0.75, 0.35],
    },
    "key": {
        "grasp_dir": "key_grasp",
        "obj_dir": "10_keys",
        "color": [0.95, 0.55, 0.0],
    },
}


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


def grasp_to_matrix(grasp_data):
    pos = np.asarray(grasp_data["position"], dtype=np.float64)
    quat = grasp_data["orientation"]
    w = quat["w"]
    x, y, z = quat["xyz"]

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    transform[:3, 3] = pos
    return transform


def load_yaml_grasps(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    grasps_dict = data.get("grasps", {})
    grasps = list(grasps_dict.items())
    return data, grasps


def choose_grasps(grasps, max_grasps, sample_mode, show_all, seed):
    if show_all or len(grasps) <= max_grasps:
        return grasps

    if sample_mode == "top":
        return sorted(
            grasps,
            key=lambda item: float(item[1].get("confidence", 0.0)),
            reverse=True,
        )[:max_grasps]

    rng = random.Random(seed)
    return rng.sample(grasps, max_grasps)


def find_obj_path(dataset_obj_dir, category, yaml_path, yaml_data):
    category_cfg = CATEGORY_CONFIG[category]
    obj_filename = os.path.basename(yaml_data.get("object_file", "")) or f"{os.path.splitext(os.path.basename(yaml_path))[0]}.obj"
    obj_filename = os.path.basename(obj_filename)

    direct_path = os.path.join(dataset_obj_dir, category_cfg["obj_dir"], obj_filename)
    if os.path.exists(direct_path):
        return direct_path

    fallback = glob.glob(os.path.join(dataset_obj_dir, "**", obj_filename), recursive=True)
    if fallback:
        return fallback[0]

    return None


def show_scene(geometries, window_title):
    action = {"value": "next"}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_title, width=1280, height=900)

    for geometry in geometries:
        vis.add_geometry(geometry)

    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True

    def make_callback(value):
        def callback(vis):
            action["value"] = value
            vis.close()
            return False
        return callback

    vis.register_key_callback(ord("Q"), make_callback("next"))
    vis.register_key_callback(ord("S"), make_callback("skip_object"))
    vis.register_key_callback(ord("C"), make_callback("skip_category"))
    vis.register_key_callback(256, make_callback("quit"))

    print("窗口快捷键: Q=下一个 | S=跳过当前物体 | C=跳过当前类别 | Esc=退出")
    vis.run()
    vis.destroy_window()
    return action["value"]


def visualize_raw_candidates(args):
    print("=" * 70)
    print("启动筛选前原始候选抓取可视化")
    print(f"类别: {', '.join(args.categories)}")
    print(f"候选来源: {args.grasps_dir}")
    print(f"物体来源: {args.dataset_obj_dir}")
    print(f"每个物体显示候选数: {'全部' if args.show_all else args.max_grasps}")
    print("=" * 70)

    skipped_categories = set()
    skipped_objects = set()
    shown = 0

    for category in args.categories:
        if category in skipped_categories:
            continue
        if category not in CATEGORY_CONFIG:
            print(f"跳过未知类别: {category}")
            continue

        category_cfg = CATEGORY_CONFIG[category]
        yaml_dir = os.path.join(args.grasps_dir, category_cfg["grasp_dir"])
        yaml_files = sorted(glob.glob(os.path.join(yaml_dir, "*.yaml")))

        if args.objects:
            wanted = set(args.objects)
            yaml_files = [
                path for path in yaml_files
                if os.path.splitext(os.path.basename(path))[0] in wanted
            ]

        print(f"\n[{category}] 找到 {len(yaml_files)} 个 YAML")

        for yaml_path in yaml_files:
            object_key = os.path.splitext(os.path.basename(yaml_path))[0]
            if object_key in skipped_objects:
                continue

            yaml_data, grasps = load_yaml_grasps(yaml_path)
            obj_path = find_obj_path(args.dataset_obj_dir, category, yaml_path, yaml_data)
            if obj_path is None:
                print(f"找不到 OBJ，跳过: {object_key}")
                continue

            chosen_grasps = choose_grasps(
                grasps=grasps,
                max_grasps=args.max_grasps,
                sample_mode=args.sample_mode,
                show_all=args.show_all,
                seed=args.seed + shown,
            )

            mesh = o3d.io.read_triangle_mesh(obj_path)
            if mesh.is_empty():
                print(f"OBJ 为空，跳过: {obj_path}")
                continue
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.72, 0.72, 0.72])

            scale = float(yaml_data.get("object_scale", 1.0))
            if scale != 1.0:
                mesh.scale(scale, center=(0, 0, 0))

            bbox = mesh.get_axis_aligned_bounding_box()
            max_extent = np.max(bbox.get_max_bound() - bbox.get_min_bound())
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1 if max_extent <= 0 else max_extent * 0.35,
                origin=[0, 0, 0],
            )

            base_length = float(yaml_data.get("base_length", 0.09188))
            color = category_cfg["color"]
            geometries = [mesh, coord_frame]

            for _, grasp_data in chosen_grasps:
                transform = grasp_to_matrix(grasp_data)
                geometries.append(
                    create_gripper_lineset(
                        transform,
                        color=color,
                        base_length=base_length,
                    )
                )

            print("\n" + "-" * 70)
            print(f"类别: {category}")
            print(f"物体: {object_key}")
            print(f"YAML: {yaml_path}")
            print(f"OBJ : {obj_path}")
            print(f"原始候选总数: {len(grasps)} | 本次显示: {len(chosen_grasps)}")
            print(f"采样模式: {'all' if args.show_all else args.sample_mode}")

            action = show_scene(
                geometries,
                f"RAW {category} | {object_key} | {len(chosen_grasps)}/{len(grasps)} candidates",
            )
            shown += 1

            if action == "skip_object":
                skipped_objects.add(object_key)
            elif action == "skip_category":
                skipped_categories.add(category)
                break
            elif action == "quit":
                print("用户退出。")
                print(f"已展示 {shown} 个物体。")
                return

    print(f"\n可视化结束，已展示 {shown} 个物体。")


def parse_args():
    parser = argparse.ArgumentParser(description="可视化筛选前的 GraspDatagen 原始候选抓取。")
    parser.add_argument("--grasps-dir", default=DEFAULT_GRASPS_DIR)
    parser.add_argument("--dataset-obj-dir", default=DEFAULT_OBJ_DIR)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["fork", "key"],
        choices=sorted(CATEGORY_CONFIG.keys()),
        help="默认只看 fork 和 key。",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="只看指定物体，例如 --objects fork_1 key_3。",
    )
    parser.add_argument("--max-grasps", type=int, default=500)
    parser.add_argument("--sample-mode", choices=["random", "top"], default="random")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="显示所有候选。每个物体约 12000 个夹爪，可能很卡。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    visualize_raw_candidates(parse_args())
