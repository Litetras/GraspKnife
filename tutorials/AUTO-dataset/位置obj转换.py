import argparse
import shutil
from pathlib import Path


DATASET_ROOT = Path("/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集")

# 原始物体 OBJ 根目录
DATASET_OBJ_ROOT = DATASET_ROOT / "dataset_obj"

# 任务导向抓取 JSON 根目录
TASK_GRASP_ROOT = DATASET_ROOT / "task_oriented_grasps_json"

# 输出为一个扁平 object_dataset：文件名和 grasp json stem 完全对应
DEFAULT_OUTPUT_OBJ_DIR = DATASET_ROOT / "converted_task_object_dataset"


CATEGORY_CONFIG = {
    "drill": {
        "obj_dir": "4_drills",
        "json_dir": "4_drills",
    },
    "pan": {
        "obj_dir": "7_pan",
        "json_dir": "7_pans",
    },
    "fork": {
        "obj_dir": "8_forks",
        "json_dir": "8_forks",
    },
    "key": {
        "obj_dir": "9_keys",
        "json_dir": "9_keys",
    },
    "spatula": {
        "obj_dir": "10_spatulas",
        "json_dir": "10_spatulas",
    },
}


def collect_source_objs(source_dir):
    source_objs = sorted(source_dir.glob("*.obj"))
    if not source_objs:
        return {}, []

    source_by_stem = {path.stem: path for path in source_objs}

    # 重要：长名字优先，避免 pan_1 错配 pan_10 / pan_11。
    source_stems = sorted(source_by_stem.keys(), key=len, reverse=True)
    return source_by_stem, source_stems


def match_source_stem(target_stem, source_stems):
    for source_stem in source_stems:
        if target_stem == source_stem or target_stem.startswith(source_stem + "_"):
            return source_stem
    return None


def copy_one_category(category_name, cfg, output_dir, overwrite=False):
    source_dir = DATASET_OBJ_ROOT / cfg["obj_dir"]
    json_dir = TASK_GRASP_ROOT / cfg["json_dir"]

    print("\n" + "=" * 70)
    print(f"📦 类别: {category_name}")
    print(f"OBJ 源目录 : {source_dir}")
    print(f"JSON 目录  : {json_dir}")

    if not source_dir.exists():
        print(f"❌ OBJ 源目录不存在，跳过: {source_dir}")
        return 0, 0, 0

    if not json_dir.exists():
        print(f"❌ JSON 目录不存在，跳过: {json_dir}")
        return 0, 0, 0

    source_by_stem, source_stems = collect_source_objs(source_dir)
    if not source_stems:
        print(f"❌ 未找到源 OBJ，跳过: {source_dir}")
        return 0, 0, 0

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"❌ 未找到任务导向 JSON，跳过: {json_dir}")
        return 0, 0, 0

    copied_count = 0
    skipped_count = 0
    failed_count = 0

    for json_path in json_files:
        target_stem = json_path.stem
        target_obj_path = output_dir / f"{target_stem}.obj"

        if target_obj_path.exists() and not overwrite:
            skipped_count += 1
            continue

        matched_stem = match_source_stem(target_stem, source_stems)
        if matched_stem is None:
            print(f"  ⚠️ [匹配失败] {json_path.name} 找不到对应源 OBJ")
            failed_count += 1
            continue

        source_obj_path = source_by_stem[matched_stem]
        shutil.copy2(source_obj_path, target_obj_path)
        copied_count += 1
        print(f"  ✅ {source_obj_path.name} -> {target_obj_path.name}")

    print(
        f"✅ {category_name} 完成: copied={copied_count}, "
        f"skipped={skipped_count}, failed={failed_count}"
    )
    return copied_count, skipped_count, failed_count


def copy_and_match_objs(categories, output_dir, overwrite=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 开始按任务导向 JSON 匹配复制 OBJ")
    print(f"输出目录: {output_dir}")
    print(f"处理类别: {', '.join(categories)}")
    print(f"覆盖已有文件: {overwrite}")
    print("=" * 70)

    total_copied = 0
    total_skipped = 0
    total_failed = 0

    for category_name in categories:
        cfg = CATEGORY_CONFIG.get(category_name)
        if cfg is None:
            print(f"⚠️ 未知类别，跳过: {category_name}")
            continue

        copied, skipped, failed = copy_one_category(
            category_name=category_name,
            cfg=cfg,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        total_copied += copied
        total_skipped += skipped
        total_failed += failed

    print("\n" + "=" * 70)
    print("🎉 OBJ 转换完成")
    print(f"✅ 新复制: {total_copied}")
    print(f"⏭️ 已存在跳过: {total_skipped}")
    print(f"❌ 匹配失败: {total_failed}")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据任务导向 grasp JSON 文件名，复制并重命名对应 OBJ。"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["drill", "pan", "fork", "key", "spatula"],
        choices=sorted(CATEGORY_CONFIG.keys()),
        help="要处理的类别。默认处理 drill/pan/fork/key/spatula。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_OBJ_DIR),
        help="输出 OBJ 目录。默认写入 converted_task_object_dataset。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果目标 OBJ 已存在，则覆盖。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    copy_and_match_objs(
        categories=args.categories,
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
    )
