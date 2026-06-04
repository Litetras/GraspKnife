import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path


DATASET_ROOT = Path("/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集")
OBJ_DIR = DATASET_ROOT / "dataset_obj" / "11_spatulas"
GRASP_DIR = DATASET_ROOT / "grasps" / "spatula_grasp"
FILTERED_JSON_DIR = DATASET_ROOT / "task_oriented_grasps_json_spatula"

ANNOTATION_JSONS = [
    Path("/home/zyp/GraspGen/spatula_dataset_boundaries_auto.json"),
    Path("/home/zyp/GraspGen/final_spatula_task_oriented_dataset.json"),
]

BAD_INDEX = 1


def natural_key(path):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(path))
    ]


def find_existing_indices(suffix, directory):
    indices = []
    for path in directory.glob(f"spatula_*.{suffix}"):
        match = re.fullmatch(r"spatula_(\d+)\." + re.escape(suffix), path.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def replace_obj_internal_name(path, new_base_name, dry_run):
    if dry_run:
        return

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    updated = []
    for line in lines:
        if line.startswith("o ") or line.startswith("g "):
            updated.append(f"o {new_base_name}\n")
        else:
            updated.append(line)
    path.write_text("".join(updated), encoding="utf-8")


def update_yaml_object_file(path, new_index, dry_run):
    if dry_run:
        return

    text = path.read_text(encoding="utf-8")
    text = re.sub(r"spatula_\d+\.obj", f"spatula_{new_index}.obj", text)
    path.write_text(text, encoding="utf-8")


def renumber_files(directory, suffix, max_index, dry_run):
    print(f"\n--- 重编号 {directory} (*.{suffix}) ---")
    bad_path = directory / f"spatula_{BAD_INDEX}.{suffix}"
    if bad_path.exists():
        print(f"删除坏样本: {bad_path}")
        if not dry_run:
            bad_path.unlink()

    existing_indices = find_existing_indices(suffix, directory)
    move_indices = [idx for idx in existing_indices if idx > BAD_INDEX and idx <= max_index]

    temp_paths = []
    for old_index in move_indices:
        old_path = directory / f"spatula_{old_index}.{suffix}"
        temp_path = directory / f".__tmp_spatula_{old_index}.{suffix}"
        print(f"临时移动: {old_path.name} -> {temp_path.name}")
        temp_paths.append((old_index, temp_path))
        if not dry_run:
            old_path.rename(temp_path)

    for old_index, temp_path in temp_paths:
        new_index = old_index - 1
        new_path = directory / f"spatula_{new_index}.{suffix}"
        print(f"重命名: {temp_path.name} -> {new_path.name}")
        if not dry_run:
            temp_path.rename(new_path)
            if suffix == "obj":
                replace_obj_internal_name(new_path, f"spatula_{new_index}", dry_run=False)
            elif suffix == "yaml":
                update_yaml_object_file(new_path, new_index, dry_run=False)


def remap_annotation_json(path, max_index, dry_run):
    print(f"\n--- 更新 JSON: {path} ---")
    if not path.exists():
        print("跳过，不存在。")
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    remapped = {}
    removed = 0
    moved = 0
    for key, value in data.items():
        match = re.fullmatch(r"spatula_(\d+)", key)
        if not match:
            remapped[key] = value
            continue

        index = int(match.group(1))
        if index == BAD_INDEX:
            print(f"删除 JSON 条目: {key}")
            removed += 1
            continue
        if BAD_INDEX < index <= max_index:
            new_key = f"spatula_{index - 1}"
            print(f"JSON key: {key} -> {new_key}")
            remapped[new_key] = value
            moved += 1
        else:
            remapped[key] = value

    print(f"JSON 统计: 删除 {removed}，重命名 {moved}，最终 {len(remapped)} 条")
    if not dry_run:
        with path.open("w", encoding="utf-8") as f:
            json.dump(remapped, f, ensure_ascii=False, indent=4)


def backup_path(path, backup_root, dry_run):
    if not path.exists():
        return

    target = backup_root / path.name
    print(f"备份: {path} -> {target}")
    if dry_run:
        return

    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)


def move_stale_filtered_results(backup_root, dry_run):
    if not FILTERED_JSON_DIR.exists():
        return

    stale_target = backup_root / f"{FILTERED_JSON_DIR.name}_stale"
    print(f"\n旧筛选结果会失效，移动到: {stale_target}")
    if dry_run:
        return

    shutil.move(str(FILTERED_JSON_DIR), str(stale_target))


def main():
    parser = argparse.ArgumentParser(
        description="删除 spatula_1，并把 spatula_2..N 连同 OBJ/YAML/JSON 标注整体前移重编号。"
    )
    parser.add_argument("--apply", action="store_true", help="真正执行修改；默认只 dry-run 打印计划。")
    parser.add_argument("--max-index", type=int, default=None, help="默认自动从 OBJ/YAML 中推断最大编号。")
    args = parser.parse_args()

    dry_run = not args.apply

    obj_indices = find_existing_indices("obj", OBJ_DIR)
    yaml_indices = find_existing_indices("yaml", GRASP_DIR)
    if not obj_indices:
        raise RuntimeError(f"找不到 spatula OBJ: {OBJ_DIR}")
    if not yaml_indices:
        raise RuntimeError(f"找不到 spatula YAML: {GRASP_DIR}")

    max_index = args.max_index or max(max(obj_indices), max(yaml_indices))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = DATASET_ROOT / f"_spatula_remove_1_backup_{timestamp}"

    print("=" * 80)
    print("删除 spatula_1 并整体前移重编号")
    print(f"dry_run: {dry_run}")
    print(f"max_index: {max_index}")
    print(f"OBJ dir: {OBJ_DIR}")
    print(f"YAML dir: {GRASP_DIR}")
    print(f"backup: {backup_root}")
    print("=" * 80)

    backup_path(OBJ_DIR, backup_root, dry_run)
    backup_path(GRASP_DIR, backup_root, dry_run)
    for json_path in ANNOTATION_JSONS:
        backup_path(json_path, backup_root, dry_run)

    renumber_files(OBJ_DIR, "obj", max_index, dry_run)
    renumber_files(GRASP_DIR, "yaml", max_index, dry_run)

    for json_path in ANNOTATION_JSONS:
        remap_annotation_json(json_path, max_index, dry_run)

    move_stale_filtered_results(backup_root, dry_run)

    print("\n" + "=" * 80)
    if dry_run:
        print("这是 dry-run，没有修改任何文件。确认无误后加 --apply 执行。")
    else:
        print("完成。接下来请重新运行智能筛选脚本生成 task_oriented_grasps_json_spatula。")
    print("=" * 80)


if __name__ == "__main__":
    main()
