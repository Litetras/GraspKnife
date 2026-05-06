import shutil
from pathlib import Path

def copy_and_match_objs():
    # 1. 源目录：包含 1_knives, 2_hammers 等分类子文件夹的基础 OBJ 总目录
    SOURCE_DIR = Path("/home/zyp/Desktop/dataset_obj")
    
    # 2. JSON 目录：存放我们刚刚用智能筛选脚本生成的、带有后缀的抓取 JSON 目录
    # （例如 /home/zyp/Desktop/task_oriented_grasps_json）
    JSON_DIR = Path("/home/zyp/Desktop/task_oriented_grasps_json")
    
    # 3. 输出目录：专门用来存放匹配复制好、并加上了语义后缀的 obj 目录
    OUTPUT_OBJ_DIR = Path("/home/zyp/Desktop/tutorial_object_dataset")

    # 确保输出目录存在，如果没有则自动创建
    OUTPUT_OBJ_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🔍 开始扫描 JSON 并生成带抓取后缀的 OBJ 模型...\n")

    # ==============================================================
    # 第一步：递归获取所有原始 obj 文件，建立基础模型库 (Base Models)
    # ==============================================================
    # rglob 会自动穿透所有子文件夹，把 knife_1.obj, hammer_1.obj 都找出来
    source_objs = list(SOURCE_DIR.rglob("*.obj"))
    if not source_objs:
        print(f"❌ 源目录 {SOURCE_DIR} 及其子文件夹中没有找到任何 .obj 文件！")
        return
        
    # 建立字典：基础模型名 -> 真实绝对路径
    # 例如: {'knife_1': Path('/home/zyp/Desktop/dataset_obj/1_knives/knife_1.obj')}
    base_name_to_path = {f.stem: f for f in source_objs}
    
    # 按名字长度降序排列（极其重要！防止 knife_10 被截断匹配成 knife_1）
    base_names = sorted(list(base_name_to_path.keys()), key=len, reverse=True)

    # ==============================================================
    # 第二步：获取所有带有后缀的 JSON 抓取文件
    # ==============================================================
    # 这里会读取类似于 knife_1_Handle_Up.json 的文件
    json_files = list(JSON_DIR.rglob("*.json"))
    if not json_files:
        print(f"❌ JSON 目录 {JSON_DIR} 中没有找到任何 .json 文件！")
        return

    success_count = 0

    # ==============================================================
    # 第三步：遍历 JSON，为每个场景量身定做一个同名（带后缀）的 OBJ
    # ==============================================================
    for json_path in json_files:
        # target_name 就是带后缀的场景名，例如: "knife_1_Handle_Up"
        target_name = json_path.stem
        
        # 我们要生成的 obj 名字也要带这个后缀，确保和 JSON 名字完全一样
        target_obj_path = OUTPUT_OBJ_DIR / f"{target_name}.obj"
        
        # 如果这个加上后缀的 obj 已经存在了，直接跳过
        if target_obj_path.exists():
            continue

        # 寻找它对应的原始无后缀模型是哪一个 (通过前缀匹配)
        matched_base = None
        for base in base_names:
            if target_name.startswith(base):
                matched_base = base
                break
                
        if matched_base:
            # 从字典中提取出原始 obj 的真实路径
            source_obj_path = base_name_to_path[matched_base]
            
            # 执行核心操作：将基础 obj 复制，并重命名为带有后缀的新 obj
            shutil.copy(source_obj_path, target_obj_path)
            
            print(f"  └── [匹配成功] {source_obj_path.name} -> {target_obj_path.name}")
            success_count += 1
        else:
            print(f"  ⚠️ [匹配失败] 找不到 JSON 文件 {target_name}.json 对应的原始基础模型！")

    print(f"\n🎉 匹配与重命名完成！")
    print(f"共成功生成了 {success_count} 个带有语义抓取后缀的 OBJ 模型。")
    print(f"📁 它们已全部扁平化存入: {OUTPUT_OBJ_DIR}")

if __name__ == "__main__":
    copy_and_match_objs()