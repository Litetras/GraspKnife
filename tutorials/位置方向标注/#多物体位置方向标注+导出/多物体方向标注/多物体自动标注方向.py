import os
import json

def merge_annotations(boundaries_json, directions_json, output_json):
    print("="*70)
    print("🧬 启动【类别级到实例级】数据集终极融合模块 (Category-to-Instance Broadcasting)")
    print("="*70)

    # 1. 检查文件是否存在
    if not os.path.exists(boundaries_json):
        print(f"❌ 找不到几何边界文件 (Instance Boundaries): {boundaries_json}")
        return
    if not os.path.exists(directions_json):
        print(f"❌ 找不到操作者方向文件 (Category Orientations): {directions_json}")
        return

    # 2. 加载两份数据
    with open(boundaries_json, 'r', encoding='utf-8') as f:
        boundaries_data = json.load(f)
        
    with open(directions_json, 'r', encoding='utf-8') as f:
        directions_data = json.load(f)

    # 3. 遍历所有 3D 实例，把类别的操作者方向“广播”给每一个实体模型
    final_dataset = {}
    missing_directions_count = 0

    for base_name, instance_info in boundaries_data.items():
        category = instance_info.get("category")
        
        # 复制原本的位置信息 (Grasp Region 的几何依据，如 split_axis, boundary_coord 等)
        final_dataset[base_name] = instance_info.copy()
        
        # 匹配对应的语义方向 (Operator-Centric Orientations)
        if category in directions_data:
            # 学术升级：将字典 key 从 intents 改为 orientations，对齐论文 Table 1
            final_dataset[base_name]["orientations"] = directions_data[category]
        else:
            print(f"⚠️ 警告: 实例 {base_name} 的类别 [{category}] 没有在方向字典中找到配置！")
            final_dataset[base_name]["orientations"] = {}
            missing_directions_count += 1

    # 4. 保存为最终的 Dataset JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=4)

    print(f"\n🎉 完美融合！共将四维规则映射到了 {len(final_dataset)} 个 3D 实例上。")
    if missing_directions_count > 0:
        print(f"⚠️ 有 {missing_directions_count} 个实例缺少方向配置。")
    print(f"📁 最终的 Task-Oriented Ground Truth 数据集已保存至: {output_json}")
    print("="*70)

if __name__ == "__main__":
    # 输入文件 1：包含 Region 切分信息的 JSON
    BOUNDARIES_JSON = "all_dataset_boundaries_auto.json"   
    # 输入文件 2：包含 Orientation 信息的 JSON (我们刚刚标出来的)
    DIRECTIONS_JSON = "category_grasp_directions.json"     
    
    # 终极输出文件 (直接用于算法训练的数据)
    OUTPUT_JSON = "final_task_oriented_dataset.json"
    
    merge_annotations(BOUNDARIES_JSON, DIRECTIONS_JSON, OUTPUT_JSON)