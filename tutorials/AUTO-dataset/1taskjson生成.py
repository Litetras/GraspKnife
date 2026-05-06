import os
import json
import re
from pathlib import Path

# ==========================================
# 配置区域
# ==========================================
# 指向你存放合并后缀后 OBJ 文件的目录
DATA_DIR = Path("/home/zyp/Desktop/zyp_dataset7_clip/tutorial/tutorial_object_dataset") 

# 输出的 JSON 文件名
OUTPUT_JSON = "task_texts.json"

def generate_task_texts():
    print("="*60)
    print("💬 启动【自然语言任务文本】自动生成脚本")
    print("="*60)

    if not DATA_DIR.exists():
        print(f"❌ 找不到数据目录: {DATA_DIR}")
        return

    # 1. 获取所有 .obj 文件
    obj_files = list(DATA_DIR.glob("*.obj"))
    if not obj_files:
        print("❌ 目录中没有找到 .obj 文件！")
        return

    print(f"🔍 找到 {len(obj_files)} 个带后缀的模型，正在提取语义...")

    task_texts = {}
    success_count = 0

    # 2. 遍历文件，提取自然语言
    for obj_path in obj_files:
        # 去掉 .obj 后缀，得到纯粹的名字 (例如: hammer_10_down_handle)
        base_name = obj_path.stem
        
        # 提取方向词 (Direction) 和 部位词 (Part)
        # 我们寻找下划线后面的这些关键词
        direction = ""
        part = ""

        # 匹配方向 (转小写处理)
        lower_name = base_name.lower()
        if "_up_" in lower_name or lower_name.endswith("_up"):
            direction = "up"
        elif "_down_" in lower_name or lower_name.endswith("_down"):
            direction = "down"
        elif "_front_" in lower_name or lower_name.endswith("_front"):
            direction = "front"
        elif "_back_" in lower_name or lower_name.endswith("_back"):
            direction = "back"
        elif "_left_" in lower_name or lower_name.endswith("_left"):
            direction = "left"
        elif "_right_" in lower_name or lower_name.endswith("_right"):
            direction = "right"

        # 匹配部位
        if "_handle" in lower_name:
            part = "handle"
        elif "_head" in lower_name:
            part = "head"
        elif "_blade" in lower_name:
            part = "blade"
        elif "_rim" in lower_name:
            part = "rim"
        elif "_shaft" in lower_name:
            part = "shaft"

        # 组合成 task_text (如果只有方向没部位，或者只有部位没方向，也能兼容)
        task_text_parts = []
        if direction:
            task_text_parts.append(direction)
        if part:
            task_text_parts.append(part)
            
        task_text = " ".join(task_text_parts)

        # 只要提取出了语义，就写入字典
        if task_text:
            task_texts[base_name] = {
                "task1": task_text
            }
            success_count += 1
        else:
            print(f"⚠️ [警告] 无法从文件名中提取语义: {base_name}")

    # 3. 按照字母顺序对字典的键进行排序 (让输出的 JSON 更好看、方便查阅)
    sorted_task_texts = {k: task_texts[k] for k in sorted(task_texts.keys())}

    # 4. 导出为格式化的 JSON 文件
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted_task_texts, f, indent=2, ensure_ascii=False)

    print("\n🎉 生成完毕！")
    print("-" * 60)
    print(f"📄 成功提取了 {success_count} 个任务文本。")
    print(f"💾 文件已保存为: {os.path.abspath(OUTPUT_JSON)}")
    print("="*60)
    print("👉 请将此文件移动到 DataLoader 期望的路径 (如 /results/tutorial/task_texts.json)！")

if __name__ == "__main__":
    generate_task_texts()