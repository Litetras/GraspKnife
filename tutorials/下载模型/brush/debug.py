import json
import os
import shutil

def merge_brush_boundaries(main_json_path, new_brush_json_path):
    print("="*70)
    print("🧬 启动【数据集边界坐标】无缝融合替换器")
    print(f"📖 主数据集: {main_json_path}")
    print(f"🖌️ 新刷子集: {new_brush_json_path}")
    print("="*70)

    # 1. 检查文件是否存在
    if not os.path.exists(main_json_path):
        print(f"❌ 找不到主文件: {main_json_path}")
        return
    if not os.path.exists(new_brush_json_path):
        print(f"❌ 找不到新刷子文件: {new_brush_json_path}")
        return

    # 2. 安全第一：为总表创建备份
    backup_path = main_json_path + ".bak"
    shutil.copy2(main_json_path, backup_path)
    print(f"🛡️ 已为您自动创建数据备份: {backup_path}")

    # 3. 加载两份 JSON 数据
    with open(main_json_path, 'r', encoding='utf-8') as f:
        main_data = json.load(f)
        
    with open(new_brush_json_path, 'r', encoding='utf-8') as f:
        new_brush_data = json.load(f)

    # 4. 遍历新刷子数据，将其覆盖注入到主数据中
    update_count = 0
    insert_count = 0

    for brush_name, brush_info in new_brush_data.items():
        if brush_name in main_data:
            # 原本存在，直接覆盖旧的边界坐标
            main_data[brush_name] = brush_info
            update_count += 1
        else:
            # 万一是新增的刷子，直接插入
            main_data[brush_name] = brush_info
            insert_count += 1

    # 5. 将融合后的数据写回主文件
    with open(main_json_path, 'w', encoding='utf-8') as f:
        json.dump(main_data, f, indent=4)

    print("\n" + "="*70)
    print(f"🎉 融合圆满成功！")
    print(f"   -> 🔄 成功覆盖替换了 {update_count} 个旧刷子模型坐标。")
    if insert_count > 0:
        print(f"   -> ➕ 新增了 {insert_count} 个此前不存在的刷子坐标。")
    print(f"📁 最新的边界真值表已保存至: {main_json_path}")
    print("="*70)

if __name__ == "__main__":
    # 配置你的两个 JSON 文件名 (确保它们在同一级目录，或写上绝对路径)
    MAIN_JSON = "all_dataset_boundaries_auto.json"
    NEW_BRUSH_JSON = "scaled_brush_boundaries_auto.json"
    
    merge_brush_boundaries(MAIN_JSON, NEW_BRUSH_JSON)