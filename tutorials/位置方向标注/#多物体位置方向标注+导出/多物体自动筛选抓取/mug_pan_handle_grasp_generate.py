import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml


DATASET_ROOT = Path("/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集")
DEFAULT_DATASET_OBJ_DIR = DATASET_ROOT / "dataset_obj"
DEFAULT_RAW_GRASPS_DIR = DATASET_ROOT / "grasps"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "handle_biased_generation"
DEFAULT_GRASPDATAGEN_ROOT = Path("/home/zyp/GraspDataGen")

DEFAULT_MUG_ANNOTATION_JSON = Path("/home/zyp/GraspGen/final_task_oriented_dataset.json")
DEFAULT_PAN_ANNOTATION_JSON = Path("/home/zyp/GraspGen/final_pan_task_oriented_dataset.json")

CATEGORY_CONFIG = {
    "mug": {
        "annotation_categories": {"7_mugs", "mugs", "mug"},
        "obj_dir": "7_mugs",
        "raw_grasp_dir": "mug_grasp",
        "handle_grasp_dir": "mug_handle_grasp",
    },
    "pan": {
        "annotation_categories": {"pans", "8_pans", "8_pan", "pan"},
        "obj_dir": "8_pan",
        "raw_grasp_dir": "pan_grasp",
        "handle_grasp_dir": "pan_handle_grasp",
    },
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def category_from_annotation(object_key, info):
    category_name = str(info.get("category", "")).lower()
    for category, cfg in CATEGORY_CONFIG.items():
        if object_key.startswith(f"{category}_"):
            return category
        if category_name in cfg["annotation_categories"]:
            return category
    return None


def load_annotations(args):
    annotations = {}
    for path in [args.mug_annotation_json, args.pan_annotation_json]:
        if not path.exists():
            print(f"⚠️ 标注文件不存在，跳过: {path}")
            continue

        for object_key, info in load_json(path).items():
            category = category_from_annotation(object_key, info)
            if category is None or category not in args.categories:
                continue
            annotations[object_key] = info

    if args.objects:
        wanted = set(args.objects)
        annotations = {k: v for k, v in annotations.items() if k in wanted}

    return dict(sorted(annotations.items()))


def find_obj_path(dataset_obj_dir, category, object_key):
    obj_name = f"{object_key}.obj"
    direct = dataset_obj_dir / CATEGORY_CONFIG[category]["obj_dir"] / obj_name
    if direct.exists():
        return direct

    matches = sorted(dataset_obj_dir.glob(f"**/{obj_name}"))
    return matches[0] if matches else None


def handle_coord_mask(coords, info, margin):
    mode = info.get("mode")
    if mode == "2_points":
        boundary = float(info["boundary_coord"])
        if bool(info["target_is_positive"]):
            return coords >= boundary - margin
        return coords <= boundary + margin

    if mode == "3_points":
        boundary_min = float(info["boundary_min"])
        boundary_max = float(info["boundary_max"])
        return (coords >= boundary_min - margin) & (coords <= boundary_max + margin)

    raise ValueError(f"未知位置标注模式: {mode}")


def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = []
        for geometry_name, geometry in mesh.geometry.items():
            transform = mesh.graph.get(geometry_name)[0]
            geom = geometry.copy()
            geom.apply_transform(transform)
            meshes.append(geom)
        mesh = trimesh.util.concatenate(meshes)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"无法读取为 Trimesh: {path}")
    if len(mesh.faces) == 0:
        raise ValueError(f"mesh 没有 face: {path}")
    return mesh


def select_handle_faces(mesh, info, margin, face_selection):
    axis = int(info["split_axis"])

    if face_selection == "centroid":
        coords = mesh.triangles_center[:, axis]
        return handle_coord_mask(coords, info, margin)

    vertex_coords = mesh.vertices[mesh.faces][:, :, axis]
    vertex_mask = handle_coord_mask(vertex_coords, info, margin)
    if face_selection == "any_vertex":
        return np.any(vertex_mask, axis=1)
    if face_selection == "all_vertices":
        return np.all(vertex_mask, axis=1)

    raise ValueError(f"未知 face-selection: {face_selection}")


def export_handle_mesh(obj_path, info, output_path, margin, face_selection):
    mesh = load_mesh(obj_path)
    face_mask = select_handle_faces(mesh, info, margin, face_selection)
    face_indices = np.flatnonzero(face_mask)

    if len(face_indices) == 0:
        return None, {
            "total_faces": len(mesh.faces),
            "handle_faces": 0,
            "handle_ratio": 0.0,
            "reason": "no_handle_faces",
        }

    handle_mesh = mesh.submesh([face_indices], append=True, repair=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle_mesh.export(output_path)

    return output_path, {
        "total_faces": len(mesh.faces),
        "handle_faces": len(face_indices),
        "handle_ratio": len(face_indices) / max(len(mesh.faces), 1),
        "reason": "ok",
    }


def run_graspdatagen(args, object_key, handle_obj_path, env_root):
    script_path = args.graspdatagen_root / "scripts/graspgen/grasp_guess.py"
    if not script_path.exists():
        raise FileNotFoundError(f"找不到 GraspDataGen 脚本: {script_path}")

    command = [
        str(args.python),
        str(script_path),
        "--gripper_config",
        args.gripper_config,
        "--object_file",
        str(handle_obj_path),
        "--num_grasps",
        str(args.num_grasps),
        "--num_orientations",
        str(args.num_orientations),
        "--percent_random_guess_angle",
        str(args.percent_random_guess_angle),
        "--num_offsets",
        str(args.num_offsets),
        "--max_guess_tries",
        str(args.max_guess_tries),
        "--device",
        args.device,
    ]

    env = os.environ.copy()
    env["GRASP_DATASET_DIR"] = str(env_root)

    print(f"   🧪 GraspDataGen: {object_key}")
    if args.print_command:
        print("   CMD:", " ".join(command))

    subprocess.run(
        command,
        cwd=str(args.graspdatagen_root),
        env=env,
        check=True,
    )

    candidates = sorted(env_root.glob(f"grasp_guess_data/**/*{object_key}.yaml"))
    if not candidates:
        candidates = sorted(env_root.glob("grasp_guess_data/**/*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"GraspDataGen 没有输出 YAML: {env_root}")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def patch_handle_yaml(handle_yaml_path, full_obj_path, handle_obj_path, output_path):
    data = load_yaml(handle_yaml_path)
    data["object_file"] = str(full_obj_path)
    data["handle_biased_candidate"] = True
    data["handle_candidate_object_file"] = str(handle_obj_path)
    data["full_object_file"] = str(full_obj_path)
    save_yaml(data, output_path)
    return output_path


def merge_yaml_grasps(original_yaml_path, handle_yaml_path, output_path):
    handle_data = load_yaml(handle_yaml_path)

    if original_yaml_path.exists():
        merged = copy.deepcopy(load_yaml(original_yaml_path))
        original_grasps = merged.get("grasps", {}) or {}
    else:
        merged = copy.deepcopy(handle_data)
        original_grasps = {}

    handle_grasps = handle_data.get("grasps", {}) or {}
    merged["created_with"] = "grasp_guess + handle_biased_generation"
    merged["handle_biased_augmented"] = True
    merged["handle_candidate_count"] = len(handle_grasps)
    merged["original_candidate_count"] = len(original_grasps)

    merged_grasps = dict(original_grasps)
    for grasp_key, grasp_value in handle_grasps.items():
        new_key = f"handle_{grasp_key}"
        suffix = 1
        while new_key in merged_grasps:
            suffix += 1
            new_key = f"handle_{grasp_key}_{suffix}"
        merged_grasps[new_key] = grasp_value

    merged["grasps"] = merged_grasps
    save_yaml(merged, output_path)
    return output_path, len(original_grasps), len(handle_grasps), len(merged_grasps)


def process_object(args, object_key, info):
    category = category_from_annotation(object_key, info)
    cfg = CATEGORY_CONFIG[category]

    full_obj_path = find_obj_path(args.dataset_obj_dir, category, object_key)
    if full_obj_path is None:
        print(f"❌ 找不到 OBJ: {object_key}")
        return False

    handle_obj_path = args.output_root / "handle_candidate_objs" / category / f"{object_key}.obj"
    handle_obj_path, stats = export_handle_mesh(
        obj_path=full_obj_path,
        info=info,
        output_path=handle_obj_path,
        margin=args.region_margin,
        face_selection=args.face_selection,
    )

    print(
        f"\n[{category}] {object_key}: handle_faces={stats['handle_faces']}/"
        f"{stats['total_faces']} ({stats['handle_ratio']:.2%})"
    )

    if handle_obj_path is None:
        print("   ⚠️ 把手区域没有 face，跳过。")
        return False

    if args.skip_graspgen:
        print(f"   ✅ 已导出 handle OBJ: {handle_obj_path}")
        return True

    handle_output_path = (
        args.output_root
        / "handle_grasps"
        / cfg["handle_grasp_dir"]
        / f"{object_key}.yaml"
    )
    if handle_output_path.exists() and not args.overwrite:
        print(f"   ⏭️ handle YAML 已存在，跳过生成: {handle_output_path}")
    else:
        env_root = args.output_root / "_graspdatagen_env" / category / object_key
        if args.overwrite and env_root.exists():
            shutil.rmtree(env_root)
        env_root.mkdir(parents=True, exist_ok=True)

        generated_yaml = run_graspdatagen(args, object_key, handle_obj_path, env_root)
        patch_handle_yaml(
            handle_yaml_path=generated_yaml,
            full_obj_path=full_obj_path,
            handle_obj_path=handle_obj_path,
            output_path=handle_output_path,
        )
        print(f"   ✅ handle YAML: {handle_output_path}")

    if args.write_augmented:
        original_yaml_path = args.raw_grasps_dir / cfg["raw_grasp_dir"] / f"{object_key}.yaml"
        augmented_output_path = (
            args.output_root
            / "augmented_grasps"
            / cfg["raw_grasp_dir"]
            / f"{object_key}.yaml"
        )
        merged_path, original_count, handle_count, total_count = merge_yaml_grasps(
            original_yaml_path=original_yaml_path,
            handle_yaml_path=handle_output_path,
            output_path=augmented_output_path,
        )
        print(
            f"   ✅ augmented YAML: {merged_path} "
            f"原始={original_count}, handle补充={handle_count}, 合计={total_count}"
        )

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="为 mug/pan 的 handle 区域单独生成 GraspDataGen 抓取候选。"
    )
    parser.add_argument("--dataset-obj-dir", type=Path, default=DEFAULT_DATASET_OBJ_DIR)
    parser.add_argument("--raw-grasps-dir", type=Path, default=DEFAULT_RAW_GRASPS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--graspdatagen-root", type=Path, default=DEFAULT_GRASPDATAGEN_ROOT)
    parser.add_argument("--mug-annotation-json", type=Path, default=DEFAULT_MUG_ANNOTATION_JSON)
    parser.add_argument("--pan-annotation-json", type=Path, default=DEFAULT_PAN_ANNOTATION_JSON)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(CATEGORY_CONFIG.keys()),
        default=["mug", "pan"],
    )
    parser.add_argument("--objects", nargs="+", default=None)

    # region_margin 会向主体方向稍微扩一点，避免把手根部被裁掉。
    parser.add_argument("--region-margin", type=float, default=0.004)
    parser.add_argument(
        "--face-selection",
        choices=["centroid", "any_vertex", "all_vertices"],
        default="centroid",
        help="centroid 更干净；any_vertex 会保留更多把手根部 face。",
    )

    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gripper-config", default="franka_panda")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-grasps", type=int, default=8000)
    parser.add_argument("--num-orientations", type=int, default=4)
    parser.add_argument("--percent-random-guess-angle", type=float, default=1.0)
    parser.add_argument("--num-offsets", type=int, default=32)
    parser.add_argument("--max-guess-tries", type=int, default=0)

    parser.add_argument("--skip-graspgen", action="store_true", help="只裁剪 handle OBJ，不运行 GraspDataGen。")
    parser.add_argument("--no-write-augmented", dest="write_augmented", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.set_defaults(write_augmented=True)
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = load_annotations(args)

    print("=" * 80)
    print("mug/pan handle-biased grasp candidate generation")
    print(f"标注物体数: {len(annotations)}")
    print(f"OBJ root: {args.dataset_obj_dir}")
    print(f"输出目录: {args.output_root}")
    print(f"类别: {', '.join(args.categories)}")
    print("=" * 80)

    ok_count = 0
    fail_count = 0
    for object_key, info in annotations.items():
        try:
            if process_object(args, object_key, info):
                ok_count += 1
            else:
                fail_count += 1
        except subprocess.CalledProcessError as exc:
            fail_count += 1
            print(f"❌ GraspDataGen 运行失败: {object_key}, returncode={exc.returncode}")
        except Exception as exc:
            fail_count += 1
            print(f"❌ 处理失败: {object_key}: {exc}")

    print("\n" + "=" * 80)
    print(f"完成: ok={ok_count}, failed={fail_count}")
    print(f"handle-only YAML: {args.output_root / 'handle_grasps'}")
    if args.write_augmented:
        print(f"augmented YAML : {args.output_root / 'augmented_grasps'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
