from pathlib import Path

def generate_obj_list_txt(object_dir, output_txt_path):
    # 1. 定义路径
    obj_dir = Path(object_dir)
    out_path = Path(output_txt_path)

    print(f"开始扫描目录: {obj_dir}")

    # 2. 扫描目录下所有的 .obj 文件
    # 使用 glob 会获取到完整路径，我们只需要文件名
    obj_files = list(obj_dir.glob("*.obj"))

    if not obj_files:
        print(f"⚠️ 警告: 在 {obj_dir} 中没有找到任何 .obj 文件！")
        return

    # 3. 提取文件名（带后缀 .obj）并进行排序
    # 排序可以确保生成的 txt 文件逻辑清晰
    obj_names = sorted([f.name for f in obj_files])

    # 4. 写入文本文件
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            for name in obj_names:
                f.write(f"{name}\n")
        
        print(f"✅ 成功生成 TXT 文件!")
        print(f"保存路径: {out_path}")
        print(f"包含对象数量: {len(obj_names)}")
        
        # 打印前 5 个预览一下
        print("\n预览前 5 个条目:")
        for name in obj_names[:5]:
            print(f"  - {name}")

    except Exception as e:
        print(f"❌ 写入文件失败: {e}")

if __name__ == "__main__":
    # ================= 配置区 =================
    # 指向你存放那堆匹配好的 obj 文件的目录
    OBJECT_DATASET_DIR = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_dataset"
    
    # 输出 txt 文件的完整路径
    # 通常你可以命名为 my_objects.txt 或 test_objects.txt
    OUTPUT_TXT = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_list.txt"
    # ==========================================

    generate_obj_list_txt(OBJECT_DATASET_DIR, OUTPUT_TXT)