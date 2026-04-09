import json
from pathlib import Path

def generate_mapping_json(dataset_dir):
    base_dir = Path(dataset_dir)
    file_map = {}

    # 1. 自动扫描目录下所有以 "_grasps.json" 结尾的文件
    json_files = list(base_dir.glob("*_grasps.json"))
    
    if not json_files:
        print(f"在 {base_dir} 下没有找到任何 _grasps.json 文件！")
        return

    # 2. 遍历并自动推导映射关系
    for json_path in json_files:
        json_filename = json_path.name  # 例如: hammer_2_up_handle_grasps.json
        
        # 掐掉 "_grasps.json" 尾巴，换成 ".obj"
        # base_name 即为 hammer_2_up_handle
        base_name = json_filename.replace("_grasps.json", "")
        obj_filename = f"{base_name}.obj"
        
        # 写入字典
        file_map[obj_filename] = json_filename

    # 3. 格式化输出到控制台
    print("========== 映射字典生成成功 ==========")
    print(json.dumps(file_map, indent=2, ensure_ascii=False))
    
    # 4. （可选）直接保存为一个 json 文件方便后续读取
    output_mapping_path = base_dir / "dataset_mapping.json"
    with open(output_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(file_map, f, indent=2, ensure_ascii=False)
    print(f"\n映射文件已自动保存至: {output_mapping_path}")

if __name__ == "__main__":
    # 指向你的数据集目录
    BASE_DIR = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_grasp_dataset"
    
    generate_mapping_json(BASE_DIR)