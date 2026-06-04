import argparse
import json
from pathlib import Path


DATASET_ROOT = Path("/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集")
DEFAULT_OBJECT_DATASET_DIR = DATASET_ROOT / "converted_task_object_dataset"
DEFAULT_TASK_GRASP_ROOT = DATASET_ROOT / "task_oriented_grasps_json"
DEFAULT_OUTPUT_JSON = DATASET_ROOT / "tutorial_language_config_10_objects.json"


CATEGORY_DIRS = [
    "1_knives",
    "2_hammers",
    "3_brushs",
    "4_drills",
    "5_spoons",
    "6_mugs",
    "7_pans",
    "8_forks",
    "9_keys",
    "10_spatulas",
]

# 文件名里真正表达任务语义的词。
# 例如 drill_12_Head_Left -> "head left"
TASK_VOCAB = {
    "up",
    "down",
    "top",
    "low",
    "lower",
    "upper",
    "handle",
    "head",
    "blade",
    "rim",
    "front",
    "back",
    "left",
    "right",
}


def extract_task_text(stem):
    words = [word.lower() for word in stem.split("_")]
    task_words = [word for word in words if word in TASK_VOCAB]
    return " ".join(task_words)


def collect_stems_from_objects(object_dir):
    obj_dir = Path(object_dir)
    obj_files = sorted(obj_dir.glob("*.obj"))
    return [path.stem for path in obj_files]


def collect_stems_from_task_jsons(task_grasp_root, category_dirs):
    root = Path(task_grasp_root)
    stems = []

    for category_dir in category_dirs:
        json_dir = root / category_dir
        if not json_dir.exists():
            print(f"⚠️ JSON 类别目录不存在，跳过: {json_dir}")
            continue

        json_files = sorted(json_dir.glob("*.json"))
        if not json_files:
            print(f"⚠️ JSON 类别目录为空，跳过: {json_dir}")
            continue

        stems.extend(path.stem for path in json_files)

    return stems


def generate_language_config(stems, output_json_path):
    unique_stems = sorted(set(stems))
    if not unique_stems:
        print("⚠️ 没有找到任何可生成语言配置的条目。")
        return

    data = {}
    empty_count = 0

    for stem in unique_stems:
        task_text = extract_task_text(stem)
        if not task_text:
            empty_count += 1
            print(f"  ⚠️ 无法从 {stem} 中提取任务词，task1 置空。")

        data[stem] = {"task1": task_text}

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("✅ 语言配置生成完成")
    print(f"保存路径: {output_path}")
    print(f"共生成: {len(data)} 条")
    print(f"空 task1: {empty_count} 条")
    print("=" * 70)

    print("\n预览前 10 条:")
    for key in list(data.keys())[:10]:
        print(f"  {key}: \"{data[key]['task1']}\"")


def parse_args():
    parser = argparse.ArgumentParser(
        description="为 10 类任务导向 OBJ/JSON 生成 tutorial_language_config。"
    )
    parser.add_argument(
        "--object-dir",
        default=str(DEFAULT_OBJECT_DATASET_DIR),
        help="转换后的 OBJ 目录。默认 converted_task_object_dataset。",
    )
    parser.add_argument(
        "--task-grasp-root",
        default=str(DEFAULT_TASK_GRASP_ROOT),
        help="任务导向 grasp JSON 根目录。",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="输出语言配置 JSON。",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "objects", "jsons"],
        default="auto",
        help=(
            "objects=扫描转换后的 OBJ；jsons=扫描 task_oriented_grasps_json；"
            "auto=优先 OBJ，若为空则使用 JSON。"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    stems = []
    if args.source in {"auto", "objects"}:
        stems = collect_stems_from_objects(args.object_dir)
        print(f"从 OBJ 目录读取到 {len(stems)} 条: {args.object_dir}")

    if args.source == "jsons" or (args.source == "auto" and not stems):
        stems = collect_stems_from_task_jsons(args.task_grasp_root, CATEGORY_DIRS)
        print(f"从任务 JSON 目录读取到 {len(stems)} 条: {args.task_grasp_root}")

    generate_language_config(stems, args.output_json)
