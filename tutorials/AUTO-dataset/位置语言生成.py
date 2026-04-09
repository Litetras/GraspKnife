import json
from pathlib import Path

def generate_language_config(object_dir, output_json_path):
    obj_dir = Path(object_dir)
    
    # 定义我们关心的“语义词汇表”
    # 只要文件名里包含这些单词，就会被提取出来作为自然语言指令
    vocab = ["up", "down", "top", "low", "handle", "head", "blade"]
    
    data = {}
    
    print(f"开始扫描目录 {obj_dir} 以生成语言配置...")
    
    obj_files = list(obj_dir.glob("*.obj"))
    if not obj_files:
        print(f"⚠️ 警告: 未在 {obj_dir} 找到任何 obj 文件！")
        return

    # 排序使得生成的 JSON 文件顺序有条理
    obj_names = sorted([f.stem for f in obj_files])

    for stem in obj_names:
        # 将文件名按 "_" 拆分成单词列表
        # 例如: "hammer_1_up_handle" -> ["hammer", "1", "up", "handle"]
        words = stem.split('_')
        
        # 提取在词汇表里的单词，并保持原有顺序
        task_text_list = [w for w in words if w in vocab]
        
        # 拼成一句话，例如 "up handle" 或 "low"
        task_text = " ".join(task_text_list)
        
        # 容错机制
        if not task_text:
            print(f"  ⚠️ 提示: 无法从 {stem} 中提取指令，设置为空。")
            task_text = ""

        # 写入字典
        data[stem] = {
            "task1": task_text
        }

    # 核心修改：保存为 JSON 文件
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 文件已成功生成！")
    print(f"保存路径: {output_json_path}")
    print(f"共生成 {len(data)} 条任务配置。")
    
    # 打印几条预览一下效果
    print("\n预览前 3 条配置:")
    for key in list(data.keys())[:3]:
        print(f"  {key}: \"{data[key]['task1']}\"")

if __name__ == "__main__":
    # 1. 之前存放那堆 OBJ 的文件夹路径
    OBJECT_DATASET_DIR = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_object_dataset"
    
    # 2. 输出的语言 JSON 文件路径 (可以直接存在数据集旁边)
    OUTPUT_JSON = "/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_language_config.json"
    
    generate_language_config(OBJECT_DATASET_DIR, OUTPUT_JSON)