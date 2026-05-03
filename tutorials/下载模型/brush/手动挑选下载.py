"""
ABO (Amazon Berkeley Objects) 刷子类3D模型 筛选 + 下载脚本
=============================================================
使用方式：
    # 第一步：只下元数据并筛选（推荐先跑这步）
    python abo_brush_downloader.py --step filter

    # 第二步：下载筛选出的3D模型
    python abo_brush_downloader.py --step download

    # 一步到位（先filter再download）
    python abo_brush_downloader.py --step all

    # 可选参数
    --output_dir ./abo_output      # 输出目录，默认 ./abo_output
    --workers 4                    # 并发下载线程数，默认 4
    --dry_run                      # 只打印会下什么，不真正下载

依赖：
    pip install boto3 tqdm
"""

import os
import json
import gzip
import argparse
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ──────────────────────────────────────────────
# 配置区：可以在这里改关键词和类目
# ──────────────────────────────────────────────
BUCKET = "amazon-berkeley-objects"

# 搜索关键词（商品名称 / 类目 / 关键词字段，任意命中即保留）
BRUSH_KEYWORDS = [
    "brush", "broom", "duster", "sweeper", "dustpan",
    "scrub", "cleaning brush", "hand brush", "dust brush",
    "paint brush",   # 如果不需要漆刷可以删掉
    # 中文关键词（部分listing含中文）
    "刷子", "扫帚", "毛刷", "清洁刷",
]

# 按 product_type 字段过滤（比关键词更精准，可留空 [] 不用）
BRUSH_PRODUCT_TYPES = [
    "CLEANING_BRUSH",
    "BRUSH",
    "BROOM",
    "DUST_PAN_AND_BRUSH_SET",
    "PAINT_BRUSH",
]

# ──────────────────────────────────────────────
# S3 工具函数
# ──────────────────────────────────────────────

def get_s3_client():
    """返回匿名S3客户端（不需要AWS账号）"""
    if not HAS_BOTO3:
        raise RuntimeError("请先安装 boto3：pip install boto3")
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )


def list_s3_prefix(s3, prefix):
    """列出某个前缀下的所有文件key"""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def download_s3_file(s3, key, local_path, dry_run=False):
    """下载单个S3文件"""
    if dry_run:
        print(f"  [dry_run] 会下载: s3://{BUCKET}/{key} -> {local_path}")
        return True
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return True  # 已存在，跳过
    try:
        s3.download_file(BUCKET, key, local_path)
        return True
    except Exception as e:
        print(f"  ✗ 下载失败 {key}: {e}")
        return False


# ──────────────────────────────────────────────
# Step 1：下载元数据并筛选
# ──────────────────────────────────────────────

def step_filter(output_dir: Path, dry_run=False):
    print("=" * 60)
    print("Step 1: 下载 listings 元数据并筛选刷子类商品")
    print("=" * 60)

    s3 = get_s3_client()
    meta_dir = output_dir / "listings_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 列出所有 listing metadata 文件
    print("\n[1/3] 列出 listings/metadata/ 下的文件...")
    keys = list_s3_prefix(s3, "listings/metadata/")
    json_gz_keys = [k for k in keys if k.endswith(".json.gz")]
    print(f"  共找到 {len(json_gz_keys)} 个listing文件")

    # 下载所有 listing 元数据（很小，总共几十MB）
    print("\n[2/3] 下载 listing 元数据...")
    iter_keys = tqdm(json_gz_keys) if HAS_TQDM else json_gz_keys
    for key in iter_keys:
        local = meta_dir / Path(key).name
        download_s3_file(s3, key, str(local), dry_run=dry_run)

    if dry_run:
        print("\n[dry_run] 跳过实际筛选，退出。")
        return

    # 解析并筛选
    print("\n[3/3] 解析并筛选刷子类商品...")
    all_items = []
    for gz_file in meta_dir.glob("*.json.gz"):
        with gzip.open(gz_file, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_items.append(json.loads(line))
                    except Exception:
                        pass

    print(f"  总商品数: {len(all_items)}")

    def is_brush(item):
        # 检查商品名
        names = item.get("item_name", [])
        name_str = " ".join(
            n.get("value", "") if isinstance(n, dict) else str(n)
            for n in names
        ).lower()

        # 检查关键词
        for kw in BRUSH_KEYWORDS:
            if kw.lower() in name_str:
                return True

        # 检查 product_type
        pt = item.get("product_type", "")
        if isinstance(pt, list):
            pt = " ".join(pt)
        if any(bpt.lower() in pt.lower() for bpt in BRUSH_PRODUCT_TYPES):
            return True

        # 检查 item_keywords 字段
        kws = item.get("item_keywords", [])
        kw_str = " ".join(
            k.get("value", "") if isinstance(k, dict) else str(k)
            for k in kws
        ).lower()
        for kw in BRUSH_KEYWORDS:
            if kw.lower() in kw_str:
                return True

        return False

    brush_items = [item for item in all_items if is_brush(item)]
    brush_with_3d = [item for item in brush_items if item.get("3dmodel_id")]

    print(f"  命中刷子关键词: {len(brush_items)} 个商品")
    print(f"  其中有3D模型:   {len(brush_with_3d)} 个")

    # 保存筛选结果
    result_path = output_dir / "brush_items_with_3d.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(brush_with_3d, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 筛选结果已保存到: {result_path}")
    print(f"  下一步运行: python {__file__} --step download")

    # 打印前5条预览
    print("\n── 预览前5条 ──")
    for item in brush_with_3d[:5]:
        names = item.get("item_name", [])
        name = names[0].get("value", "N/A") if names else "N/A"
        print(f"  [{item.get('3dmodel_id')}] {name}")


# ──────────────────────────────────────────────
# Step 2：下载3D模型
# ──────────────────────────────────────────────

def download_one_model(args):
    s3, model_id, local_path, dry_run = args
    key = f"3dmodels/original/{model_id}.glb"
    return download_s3_file(s3, key, local_path, dry_run=dry_run)


def step_download(output_dir: Path, workers=4, dry_run=False):
    print("=" * 60)
    print("Step 2: 下载筛选出的3D模型 (.glb)")
    print("=" * 60)

    result_path = output_dir / "brush_items_with_3d.json"
    if not result_path.exists():
        print(f"✗ 找不到筛选结果文件: {result_path}")
        print("  请先运行: python abo_brush_downloader.py --step filter")
        return

    with open(result_path, encoding="utf-8") as f:
        brush_items = json.load(f)

    print(f"\n共 {len(brush_items)} 个模型待下载")

    models_dir = output_dir / "3d_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    s3 = get_s3_client()

    # 构造任务列表
    tasks = []
    for item in brush_items:
        model_id = item["3dmodel_id"]
        local_path = str(models_dir / f"{model_id}.glb")
        tasks.append((s3, model_id, local_path, dry_run))

    # 并发下载
    success, fail = 0, 0
    iter_tasks = tqdm(tasks, desc="下载中") if HAS_TQDM else tasks

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one_model, t): t for t in tasks}
        if HAS_TQDM:
            bar = tqdm(as_completed(futures), total=len(futures), desc="下载中")
        else:
            bar = as_completed(futures)
        for fut in bar:
            if fut.result():
                success += 1
            else:
                fail += 1

    print(f"\n✓ 下载完成！成功: {success}  失败: {fail}")
    print(f"  模型保存在: {models_dir}")

    # 同时保存一份 model_id -> 商品信息 的映射
    mapping_path = output_dir / "model_id_to_info.json"
    mapping = {
        item["3dmodel_id"]: {
            "item_id": item.get("item_id"),
            "name": (item.get("item_name") or [{}])[0].get("value", ""),
            "product_type": item.get("product_type", ""),
            "brand": (item.get("brand") or [{}])[0].get("value", ""),
        }
        for item in brush_items
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"  模型信息映射: {mapping_path}")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ABO刷子3D模型下载器")
    parser.add_argument(
        "--step",
        choices=["filter", "download", "all"],
        default="all",
        help="filter=只筛选元数据, download=只下载模型, all=两步都做",
    )
    parser.add_argument(
        "--output_dir",
        default="./abo_output",
        help="输出根目录（默认 ./abo_output）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发下载线程数（默认 4）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印操作，不真正下载",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_BOTO3:
        print("✗ 缺少依赖，请先安装：pip install boto3 tqdm")
        return

    if args.step in ("filter", "all"):
        step_filter(output_dir, dry_run=args.dry_run)

    if args.step in ("download", "all"):
        step_download(output_dir, workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()