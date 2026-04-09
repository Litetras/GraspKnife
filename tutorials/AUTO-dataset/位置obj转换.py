import shutil
from pathlib import Path

def copy_and_match_objs():
    # 1. 源目录：存放最原始的 hammer_1.obj, kitchen_knife_2.obj 等
    SOURCE_DIR = Path("/home/zyp/Desktop/cleaned_aligned")
    
    # 2. JSON 目录：存放 _grasps.json 的目录（仅用来读取文件名作为参照）
    JSON_DIR = Path("/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_grasp_dataset")
    
    # 3. 输出目录：专门用来存放匹配复制好的 obj 文件的目录 【修改的核心处】
    OUTPUT_OBJ_DIR = Path("/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_dataset")

    # 确保输出目录存在，如果没有则自动创建
    OUTPUT_OBJ_DIR.mkdir(parents=True, exist_ok=True)

    print(f"开始扫描 JSON 并匹配复制 OBJ...\n")

    # 获取所有原始的 obj 文件名，作为基础模型库 (例如 ['hammer_1', 'hammer_2', ...])
    source_objs = list(SOURCE_DIR.glob("*.obj"))
    if not source_objs:
        print(f"源目录 {SOURCE_DIR} 中没有找到任何 .obj 文件！")
        return
        
    # 按名字长度降序排列（非常重要！防止 kitchen_knife_10 匹配到 kitchen_knife_1）
    base_names = sorted([f.stem for f in source_objs], key=len, reverse=True)

    # 获取 JSON 目录下所有的 _grasps.json
    json_files = list(JSON_DIR.glob("*_grasps.json"))
    if not json_files:
        print(f"JSON 目录 {JSON_DIR} 中没有找到任何 _grasps.json 文件！请先运行转换脚本。")
        return

    success_count = 0

    # 遍历每一个 json 文件，为它“量身定做”一个 obj
    for json_path in json_files:
        # 获取需要匹配的名字：去掉 "_grasps.json"
        # 例如: hammer_27_down_head_grasps.json -> hammer_27_down_head
        target_name = json_path.name.replace("_grasps.json", "")
        
        # 【修改的核心处】生成的 obj 路径指向专门的 object_dataset 文件夹
        target_obj_path = OUTPUT_OBJ_DIR / f"{target_name}.obj"
        
        # 如果这个 obj 已经存在了，跳过
        if target_obj_path.exists():
            continue

        # 寻找它对应的原始模型是哪一个
        matched_base = None
        for base in base_names:
            if target_name.startswith(base):
                matched_base = base
                break
                
        if matched_base:
            source_obj_path = SOURCE_DIR / f"{matched_base}.obj"
            
            # 执行复制并重命名，存入 OUTPUT_OBJ_DIR
            shutil.copy(source_obj_path, target_obj_path)
            print(f"  └── [匹配成功] {matched_base}.obj -> {target_obj_path.name}")
            success_count += 1
        else:
            print(f"  ⚠️ [匹配失败] 找不到 {target_name} 对应的原模型！")

    print(f"\n匹配完成！共成功生成了 {success_count} 个对应的 obj 文件。")
    print(f"文件已全部存入: {OUTPUT_OBJ_DIR}")

if __name__ == "__main__":
    copy_and_match_objs()