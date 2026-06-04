import os
import json


def merge_pan_annotations(boundaries_json, directions_json, output_json):
    print("=" * 70)
    print("🧬 启动 Pan 专属【类别级到实例级】数据集融合模块")
    print("=" * 70)

    # 1. 检查文件是否存在
    if not os.path.exists(boundaries_json):
        print(f"❌ 找不到 Pan 几何边界文件: {boundaries_json}")
        return

    if not os.path.exists(directions_json):
        print(f"❌ 找不到 Pan 操作者方向文件: {directions_json}")
        return

    # 2. 加载两份数据
    with open(boundaries_json, "r", encoding="utf-8") as f:
        boundaries_data = json.load(f)

    with open(directions_json, "r", encoding="utf-8") as f:
        directions_data = json.load(f)

    # 3. 只融合 Pan / pans 类别
    final_dataset = {}
    missing_directions_count = 0
    skipped_non_pan_count = 0

    for base_name, instance_info in boundaries_data.items():
        category = instance_info.get("category")

        # Pan 专属：只接受 category == "pans"
        if category != "pans":
            skipped_non_pan_count += 1
            continue

        # 复制实例级几何边界信息
        final_dataset[base_name] = instance_info.copy()

        # 把类别级方向 orientations 广播给每个 pan 实例
        if category in directions_data:
            final_dataset[base_name]["orientations"] = directions_data[category]
        else:
            print(f"⚠️ 警告: Pan 实例 {base_name} 没有找到类别方向配置 [{category}]")
            final_dataset[base_name]["orientations"] = {}
            missing_directions_count += 1

    # 4. 保存最终 Pan Dataset JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=4)

    print(f"\n🎉 Pan 数据融合完成！")
    print(f"✅ 成功融合 Pan 实例数量: {len(final_dataset)}")

    if skipped_non_pan_count > 0:
        print(f"⏭️ 跳过非 Pan 实例数量: {skipped_non_pan_count}")

    if missing_directions_count > 0:
        print(f"⚠️ 有 {missing_directions_count} 个 Pan 实例缺少方向配置。")

    print(f"📁 Pan 最终 Task-Oriented Ground Truth 已保存至: {output_json}")
    print("=" * 70)


if __name__ == "__main__":
    # 输入文件 1：Pan 几何边界标注文件
    BOUNDARIES_JSON = "pan_dataset_boundaries_auto.json"

    # 输入文件 2：Pan 类别方向标注文件
    DIRECTIONS_JSON = "pan_category_grasp_directions.json"

    # 输出文件：Pan 专属最终数据集
    OUTPUT_JSON = "final_pan_task_oriented_dataset.json"

    merge_pan_annotations(
        BOUNDARIES_JSON,
        DIRECTIONS_JSON,
        OUTPUT_JSON
    )