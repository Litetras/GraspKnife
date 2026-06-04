import argparse
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_OBJECT_DATASET_DIR = Path(
    "/home/zyp/Desktop/zyp_dataset7_clip/tutorial/tutorial_object_dataset"
)
DEFAULT_TRAIN_TXT = DEFAULT_OBJECT_DATASET_DIR / "train.txt"
DEFAULT_VALID_TXT = DEFAULT_OBJECT_DATASET_DIR / "valid.txt"


def get_object_instance_key(obj_name):
    """
    drill_12_Head_Left.obj -> drill_12

    同一个物体实例的不同任务抓取必须进入同一个 split，
    否则 train 里见过 drill_12_Handle_Up，valid 又测 drill_12_Head_Left，
    会让验证集不干净。
    """
    stem = Path(obj_name).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return "_".join(parts[:2])
    return stem


def get_category_name(instance_key):
    return instance_key.split("_")[0]


def split_instances_by_category(instance_keys, valid_ratio, seed):
    rng = random.Random(seed)
    by_category = defaultdict(list)

    for key in sorted(instance_keys):
        by_category[get_category_name(key)].append(key)

    train_instances = set()
    valid_instances = set()

    for category, keys in sorted(by_category.items()):
        keys = sorted(keys)
        rng.shuffle(keys)

        # 每个类别都按同一比例划分。只要类别里多于 1 个实例，就至少留 1 个验证实例。
        valid_count = round(len(keys) * valid_ratio)
        if len(keys) > 1:
            valid_count = max(1, min(valid_count, len(keys) - 1))
        else:
            valid_count = 0

        valid_part = set(keys[:valid_count])
        train_part = set(keys[valid_count:])

        valid_instances.update(valid_part)
        train_instances.update(train_part)

        print(
            f"{category:8s} objects={len(keys):3d} "
            f"train={len(train_part):3d} valid={len(valid_part):3d}"
        )

    return train_instances, valid_instances


def write_split(path, names):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for name in sorted(names):
            f.write(f"{name}\n")


def generate_train_valid_split(
    object_dir,
    train_txt,
    valid_txt,
    valid_ratio=0.2,
    seed=7,
):
    obj_dir = Path(object_dir)
    train_path = Path(train_txt)
    valid_path = Path(valid_txt)

    print("=" * 70)
    print("🚀 重新生成 train.txt / valid.txt")
    print(f"OBJ 目录: {obj_dir}")
    print(f"valid_ratio: {valid_ratio}")
    print(f"seed: {seed}")
    print("=" * 70)

    obj_files = sorted(obj_dir.glob("*.obj"))
    if not obj_files:
        print(f"❌ 未找到 OBJ: {obj_dir}")
        return

    instance_to_objs = defaultdict(list)
    for path in obj_files:
        instance_to_objs[get_object_instance_key(path.name)].append(path.name)

    train_instances, valid_instances = split_instances_by_category(
        instance_keys=instance_to_objs.keys(),
        valid_ratio=valid_ratio,
        seed=seed,
    )

    train_names = []
    valid_names = []
    for instance_key, obj_names in instance_to_objs.items():
        if instance_key in valid_instances:
            valid_names.extend(obj_names)
        else:
            train_names.extend(obj_names)

    train_names = sorted(train_names)
    valid_names = sorted(valid_names)

    write_split(train_path, train_names)
    write_split(valid_path, valid_names)

    overlap = set(train_names) & set(valid_names)
    all_names = set(train_names) | set(valid_names)

    print("\n" + "=" * 70)
    print("✅ 划分完成")
    print(f"总 OBJ 样本: {len(obj_files)}")
    print(f"物体实例数: {len(instance_to_objs)}")
    print(f"train 样本: {len(train_names)}")
    print(f"valid 样本: {len(valid_names)}")
    print(f"train/valid overlap: {len(overlap)}")
    print(f"未覆盖 OBJ: {len(set(path.name for path in obj_files) - all_names)}")
    print(f"train.txt: {train_path}")
    print(f"valid.txt: {valid_path}")
    print("=" * 70)

    print("\ntrain 预览:")
    for name in train_names[:8]:
        print(f"  {name}")

    print("\nvalid 预览:")
    for name in valid_names[:8]:
        print(f"  {name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="按类别等比例、按物体实例分组生成 train.txt / valid.txt。"
    )
    parser.add_argument("--object-dir", default=str(DEFAULT_OBJECT_DATASET_DIR))
    parser.add_argument("--train-txt", default=str(DEFAULT_TRAIN_TXT))
    parser.add_argument("--valid-txt", default=str(DEFAULT_VALID_TXT))
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_train_valid_split(
        object_dir=args.object_dir,
        train_txt=args.train_txt,
        valid_txt=args.valid_txt,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
