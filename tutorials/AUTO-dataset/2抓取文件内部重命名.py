import json
import shutil
from pathlib import Path

def fix_json_internal_names():
    print("="*60)
    print("🔧 启动【JSON 内部文件指针】同步修复脚本 (安全另存版)")
    print("="*60)

    # 1. 源目录：存在问题的旧 JSON 目录
    JSON_DIR = Path("/home/zyp/Desktop/zyp_dataset7_clip/tutorial/tutorial_grasp_dataset")
    
    # 2. 输出目录：修复后另存为的新 JSON 目录（不覆盖原文件）
    # 你可以自己修改这个路径，这里我加了 _fixed 后缀
    OUTPUT_JSON_DIR = Path("/home/zyp/Desktop/zyp_dataset7_clip/tutorial/tutorial_grasp_dataset_fixed")

    if not JSON_DIR.exists():
        print(f"❌ 找不到 JSON 目录: {JSON_DIR}")
        return

    # 创建输出目录，如果不存在会自动创建
    OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(JSON_DIR.rglob("*.json"))
    if not json_files:
        print(f"❌ 目录中没有找到 .json 文件！")
        return

    fixed_count = 0
    copied_count = 0

    for json_path in json_files:
        # 获取文件名，不带后缀 (例如: "hammer_12_Head_Up")
        target_name = json_path.stem
        expected_obj_name = f"{target_name}.obj"
        
        # 设定新 JSON 文件的保存路径
        out_json_path = OUTPUT_JSON_DIR / json_path.name

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            is_modified = False
            # 如果存在 object.file 字段且不等于预期名字，则修改
            if "object" in data and "file" in data["object"]:
                current_obj_name = data["object"]["file"]
                
                if current_obj_name != expected_obj_name:
                    data["object"]["file"] = expected_obj_name
                    is_modified = True
                    
            # 无论是否进行了修改，都将最终的 data 字典写入到新文件夹
            # 这样保证新文件夹里是一个完整的数据集
            with open(out_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            if is_modified:
                fixed_count += 1
            else:
                copied_count += 1
                
        except Exception as e:
            print(f"⚠️ 处理文件 {json_path.name} 时出错: {e}")

    print("\n🎉 修复并另存完成！")
    print(f"   ➤ 成功修复并保存了 {fixed_count} 个指向错误的 JSON。")
    print(f"   ➤ 直接同步保存了 {copied_count} 个本就正确的 JSON。")
    print(f"📁 完美、干净的数据集已全部存入:\n   {OUTPUT_JSON_DIR}")
    print("="*60)
    print("🚨 关键提醒：请务必去配置文件里，把 DataLoader 的 `grasp_root_dir` 指向这个带 _fixed 的新文件夹！")

if __name__ == "__main__":
    fix_json_internal_names()