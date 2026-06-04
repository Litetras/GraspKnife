import json
import os


TARGET_CATEGORIES = {"11_spatulas"}


def merge_spatula_annotations(boundaries_json, directions_json, output_json):
    print("=" * 70)
    print("🧬 启动 Spatula【类别级方向 -> 实例级】数据集融合模块")
    print("=" * 70)

    if not os.path.exists(boundaries_json):
        print(f"❌ 找不到几何边界文件: {boundaries_json}")
        return

    if not os.path.exists(directions_json):
        print(f"❌ 找不到操作者方向文件: {directions_json}")
        return

    with open(boundaries_json, "r", encoding="utf-8") as f:
        boundaries_data = json.load(f)

    with open(directions_json, "r", encoding="utf-8") as f:
        directions_data = json.load(f)

    final_dataset = {}
    missing_directions_count = 0
    skipped_count = 0

    for base_name, instance_info in boundaries_data.items():
        category = instance_info.get("category")

        if category not in TARGET_CATEGORIES:
            skipped_count += 1
            continue

        final_dataset[base_name] = instance_info.copy()

        if category in directions_data:
            final_dataset[base_name]["orientations"] = directions_data[category]
        else:
            print(f"⚠️ 警告: 实例 {base_name} 没有找到类别方向配置 [{category}]")
            final_dataset[base_name]["orientations"] = {}
            missing_directions_count += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=4)

    print("\n🎉 Spatula 数据融合完成！")
    print(f"✅ 成功融合实例数量: {len(final_dataset)}")

    if skipped_count > 0:
        print(f"⏭️ 跳过非 Spatula 实例数量: {skipped_count}")

    if missing_directions_count > 0:
        print(f"⚠️ 有 {missing_directions_count} 个实例缺少方向配置。")

    print(f"📁 Spatula 最终 Task-Oriented Ground Truth 已保存至: {output_json}")
    print("=" * 70)


if __name__ == "__main__":
    BOUNDARIES_JSON = "spatula_dataset_boundaries_auto.json"
    DIRECTIONS_JSON = "spatula_category_grasp_directions.json"
    OUTPUT_JSON = "final_spatula_task_oriented_dataset.json"

    merge_spatula_annotations(
        BOUNDARIES_JSON,
        DIRECTIONS_JSON,
        OUTPUT_JSON,
    )
