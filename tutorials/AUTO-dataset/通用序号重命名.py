import os
import glob
import re

# ==========================================
# 配置区域
# ==========================================
RENAME_JOBS = [
    {
        "work_dir": "/home/zyp/Desktop/objaverse_dataset/spatulas",
        "prefix": "spatula",
    },

]

# ==========================================
# 工具函数
# ==========================================
def extract_number(filename):
    """使用文件名里的数字做自然排序，保证 2 排在 10 前面。"""
    numbers = re.findall(r'\d+', os.path.basename(filename))
    return int(numbers[-1]) if numbers else 0


def rename_obj_files(work_dir, prefix):
    output_dir = os.path.join(work_dir, "renamed_output")

    if not os.path.exists(work_dir):
        raise FileNotFoundError(f"找不到文件夹: {work_dir}")

    os.makedirs(output_dir, exist_ok=True)
    obj_files = sorted(glob.glob(os.path.join(work_dir, "*.obj")), key=extract_number)

    if not obj_files:
        print(f"在 {work_dir} 中没有找到 .obj 文件，跳过。")
        return 0, output_dir

    print("=" * 60)
    print(f"目录: {work_dir}")
    print(f"共找到 {len(obj_files)} 个 OBJ 文件，准备以前缀 '{prefix}' 重命名...")

    converted_count = 0
    for index, old_path in enumerate(obj_files, start=1):
        new_base_name = f"{prefix}_{index}"
        new_filename = f"{new_base_name}.obj"
        new_path = os.path.join(output_dir, new_filename)

        try:
            with open(old_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            with open(new_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    if line.startswith('o ') or line.startswith('g '):
                        f.write(f"o {new_base_name}\n")
                    else:
                        f.write(line)

            converted_count += 1
            print(f"[{index}/{len(obj_files)}] 转换成功: {os.path.basename(old_path)} -> {new_filename}")

        except Exception as exc:
            print(f"处理文件 {old_path} 时出错: {exc}")

    print(f"✅ 完成: {converted_count}/{len(obj_files)} 个，新模型已存放在: {output_dir}")
    return converted_count, output_dir


def main():
    total_converted = 0
    output_dirs = []

    for job in RENAME_JOBS:
        converted_count, output_dir = rename_obj_files(job["work_dir"], job["prefix"])
        total_converted += converted_count
        output_dirs.append(output_dir)

    print("=" * 60)
    print(f"全部重命名完成！总计转换 {total_converted} 个 OBJ。")
    print("输出目录:")
    for output_dir in output_dirs:
        print(f"  - {output_dir}")
    print("现在你可以把这些 renamed_output 文件夹里的模型拖入 Blender 中。")
    print("=" * 60)


if __name__ == "__main__":
    main()
