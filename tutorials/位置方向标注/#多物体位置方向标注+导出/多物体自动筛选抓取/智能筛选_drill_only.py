import argparse
import importlib.util
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
FULL_FILTER_SCRIPT = CURRENT_DIR / "智能筛选.py"

DATASET_ROOT = Path("/home/zyp/pan1/#LODGrasp核心权重与数据集/7个物体数据集")
DATASET_JSON = Path("/home/zyp/GraspGen/final_task_oriented_dataset.json")
DRILL_YAML_DIR = DATASET_ROOT / "grasps" / "drill_grasp"
OUTPUT_ROOT = DATASET_ROOT / "task_oriented_grasps_json"
DRILL_OUTPUT_DIR = OUTPUT_ROOT / "4_drills"


def load_full_filter_module():
    spec = importlib.util.spec_from_file_location("smart_filter_full", FULL_FILTER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_drill_outputs():
    # 只清理 drill 的旧导出结果，避免旧规则留下的 JSON 和新规则混在一起。
    if not DRILL_OUTPUT_DIR.exists():
        return 0

    removed = 0
    for json_path in DRILL_OUTPUT_DIR.glob("*.json"):
        json_path.unlink()
        removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(description="只重新筛选 Drill 的任务导向抓取。")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="运行前只删除 task_oriented_grasps_json/4_drills 下的旧 drill JSON。",
    )
    args = parser.parse_args()

    if args.clean:
        removed = clean_drill_outputs()
        print(f"🧹 已清理旧 drill JSON: {removed} 个")

    module = load_full_filter_module()

    # 只把 drill_grasp 目录交给原筛选函数，因此不会重新处理其它 6 类物体。
    module.filter_and_convert_grasps(
        str(DATASET_JSON),
        str(DRILL_YAML_DIR),
        str(OUTPUT_ROOT),
    )


if __name__ == "__main__":
    main()
