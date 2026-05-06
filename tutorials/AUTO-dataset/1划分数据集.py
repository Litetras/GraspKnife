import os
import random
import re
from collections import defaultdict
from pathlib import Path

# ==========================================
# 配置区域
# ==========================================
# 指向你存放合并后缀后 OBJ 文件的目录
DATA_DIR = Path("/home/zyp/Desktop/zyp_dataset7_clip/tutorial/tutorial_object_dataset") 

# 训练集比例 (0.8 表示 80% 训练，20% 验证)
TRAIN_RATIO = 0.8 

# 输出的文件名
TRAIN_TXT = "train.txt"
VALID_TXT = "valid.txt"

# 随机种子，确保每次运行划分结果一致，方便复现
random.seed(42) 

def split_dataset():
    print("="*60)
    print("🔪 启动【防泄露 + 类别均衡】数据集智能划分脚本")
    print("="*60)

    if not DATA_DIR.exists():
        print(f"❌ 找不到数据目录: {DATA_DIR}")
        return

    # 1. 获取所有 .obj 文件名
    files = [f.name for f in DATA_DIR.glob("*.obj")]
    if not files:
        print("❌ 目录中没有找到 .obj 文件！")
        return

    # 2. 提取基础物体实例 (Instance)
    # 目的：将 hammer_5_up_handle 和 hammer_5_down_head 分到同一实例组 hammer_5
    object_groups = defaultdict(list)

    for f in files:
        match = re.match(r"(.*?)(_up_|_down_|_top_|_low_|_front_|_back_|_left_|_right_)", f, flags=re.IGNORECASE)
        if match:
            base_name = match.group(1)
        else:
            parts = f.replace('.obj', '').split('_')
            if len(parts) >= 3:
                base_name = '_'.join(parts[:-2])
            else:
                base_name = f.replace('.obj', '')
                
        object_groups[base_name].append(f)

    # 3. 将基础物体实例按“大类 (Category)”进行归类
    # 目的：将 hammer_5, hammer_6 归入 hammer 大类
    category_groups = defaultdict(list)
    
    for base_name in object_groups.keys():
        # 通过正则去掉末尾的数字及其前面的下划线，提取纯类别名 (例如: hammer_5 -> hammer)
        category = re.sub(r'_\d+$', '', base_name)
        category_groups[category].append(base_name)

    train_bases = []
    valid_bases = []

    print("📊 各类别划分统计 (8:2 分层抽样):")
    # 4. 在每个大类内部独立进行 8:2 划分 (Stratified Split)
    for category, bases in category_groups.items():
        random.shuffle(bases)
        
        # 计算切分点 (保证如果数据极少，至少有1个进训练集防报错)
        split_idx = max(1, int(len(bases) * TRAIN_RATIO)) if len(bases) > 1 else 1
        
        c_train = bases[:split_idx]
        c_valid = bases[split_idx:]
        
        train_bases.extend(c_train)
        valid_bases.extend(c_valid)
        
        print(f"   ➤ {category:<15} | 总实例: {len(bases):<3} | 训练集: {len(c_train):<3} | 验证集: {len(c_valid):<3}")

    # 5. 根据划分好的实例，收集它们对应的所有子任务文件名 (up/down/...)
    train_files = []
    for base in train_bases:
        train_files.extend(object_groups[base])

    valid_files = []
    for base in valid_bases:
        valid_files.extend(object_groups[base])

    # 排序让输出文件看起来更整洁
    train_files.sort()
    valid_files.sort()

    # 6. 写入到 txt 文件
    with open(TRAIN_TXT, "w", encoding="utf-8") as f:
        for item in train_files:
            f.write(f"{item}\n")

    with open(VALID_TXT, "w", encoding="utf-8") as f:
        for item in valid_files:
            f.write(f"{item}\n")

    print("\n🎉 数据集分层均衡划分圆满完成！")
    print("-" * 60)
    print(f"📦 物理模型总数 (Instances): {len(train_bases) + len(valid_bases)}")
    print(f"   ┣ 🎓 训练集分配了: {len(train_bases)} 个物体实例")
    print(f"   ┗ 🧪 验证集分配了: {len(valid_bases)} 个物体实例")
    print("-" * 60)
    print(f"📄 抓取任务总数 (Files)  : {len(files)}")
    print(f"   ┣ 🎓 train.txt 写入了: {len(train_files)} 行")
    print(f"   ┗ 🧪 valid.txt 写入了: {len(valid_files)} 行")
    print("="*60)
    print(f"请将 {TRAIN_TXT} 和 {VALID_TXT} 移动到你的 DataLoader `root_dir` 配置路径下！")

if __name__ == "__main__":
    split_dataset()