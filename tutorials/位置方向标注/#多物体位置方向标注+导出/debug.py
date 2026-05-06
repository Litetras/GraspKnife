import json
import os
import shutil

# ==========================================
# 配置区域
# ==========================================
# 填写你刚才那个边界 JSON 文件的路径
JSON_FILE_PATH = "all_dataset_boundaries_auto.json" 

def update_json_keys(json_path):
    print("="*60)
    print("🧬 启动【JSON 键名同步修复】脚本")
    print(f"📄 目标文件: {json_path}")
    print("="*60)

    if not os.path.exists(json_path):
        print(f"❌ 找不到文件: {json_path}")
        return

    # 1. 安全第一：创建备份
    backup_path = json_path + ".bak_rename"
    shutil.copy2(json_path, backup_path)
    print(f"🛡️ 已为您自动创建数据备份: {backup_path}")

    # 2. 读取原始数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. 创建一个新的字典来存储修改后的键值对
    new_data = {}
    rename_count = 0

    for old_key, value_dict in data.items():
        new_key = old_key
        
        # 匹配并替换刀的名称
        if "kitchen_knife_" in old_key:
            new_key = old_key.replace("kitchen_knife_", "knife_")
            rename_count += 1
            
        # 匹配并替换刷子的名称
        elif "aligned_brush_" in old_key:
            new_key = old_key.replace("aligned_brush_", "brush_")
            rename_count += 1
            
        # 将数据存入新键名下
        new_data[new_key] = value_dict

    # 4. 写回 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4)

    print("\n" + "="*60)
    print(f"🎉 修复完成！成功将 {rename_count} 个旧键名更新为新名字。")
    print("="*60)

if __name__ == "__main__":
    update_json_keys(JSON_FILE_PATH)