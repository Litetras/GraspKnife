import os
import trimesh
import glob

# ==========================================
# 类别规则与路径配置区域
# ==========================================
BASE_DIR = "/home/zyp/pan1/objaverse_dataset_5"

# 针对不同物体设定的几何比例验证函数
# L(最长边), W(中间边), H(最短边)
TASKS = {
    # "mugs": {
    #     "input_dir": os.path.join(BASE_DIR, "mugs"),
    #     "output_dir": os.path.join(BASE_DIR, "mugs_cleaned"),
    #     # 马克杯特征：长宽比适中，不能太扁。
    #     # 规则：最长边不能超过中间边的 2.5 倍；最薄的一边不能小于中间边的 30%
    #     "check": lambda L, W, H: (L <= 2.5 * W) and (H >= 0.3 * W)
    # },
    "spoons": {
        "input_dir": os.path.join(BASE_DIR, "spoons"),
        "output_dir": os.path.join(BASE_DIR, "spoons_cleaned"),
        # 勺子特征：细长，且扁平。
        # 规则：最长边必须大于中间边的 2.0 倍；最薄的一边不能超过中间边的 80% (容忍一些深底汤勺)
        "check": lambda L, W, H: (L >= 2.0 * W) and (H <= 0.8 * W)
    }
#     "electric_drills": {
#         "input_dir": os.path.join(BASE_DIR, "electric_drills"),
#         "output_dir": os.path.join(BASE_DIR, "electric_drills_cleaned"),
#         # 电钻特征：长宽差不多(L或T型)，厚度适中。
#         # 规则：最长边不能超过中间边的 3.5 倍(防止带超长钻头)；厚度要在中间边的 15% 到 85% 之间(排除极薄的铁片或圆球)
#         "check": lambda L, W, H: (L <= 3.5 * W) and (0.15 * W <= H <= 0.85 * W)
#     }
}

# ==========================================
# 核心清洗流程
# ==========================================
for category, config in TASKS.items():
    in_dir = config["input_dir"]
    out_dir = config["output_dir"]
    check_func = config["check"]

    if not os.path.exists(in_dir):
        print(f"⚠️ 找不到输入目录: {in_dir}，跳过 {category}。")
        continue

    os.makedirs(out_dir, exist_ok=True)
    obj_files = glob.glob(os.path.join(in_dir, "*.obj"))
    print("\n" + "="*60)
    print(f"🚀 开始基于几何形状清洗 [{category}] (共 {len(obj_files)} 个文件)")
    print("="*60)

    count = 0
    dropped_count = 0

    for file_path in obj_files:
        filename = os.path.basename(file_path)
        try:
            # 1. 读取本地 OBJ 文件
            scene_or_mesh = trimesh.load(file_path, force='mesh')
            
            # 2. 网格合并
            if isinstance(scene_or_mesh, trimesh.Scene):
                if len(scene_or_mesh.geometry) == 0:
                    dropped_count += 1
                    continue
                mesh = trimesh.util.concatenate(tuple(scene_or_mesh.geometry.values()))
            else:
                mesh = scene_or_mesh

            # 3. 居中到坐标原点
            mesh.apply_translation(-mesh.centroid)
            
            # ==================================
            # 4. 【核心：提取并验证包围盒特征】
            # ==================================
            extents = sorted(mesh.extents, reverse=True)
            length = extents[0] # L: 最长边
            width = extents[1]  # W: 中间边
            height = extents[2] # H: 最短边 (最薄处)

            # 异常极小值过滤 (过滤损坏的微小点集)
            if length <= 0.0001:
                dropped_count += 1
                continue

            # 使用该类别专属的几何规则进行判断
            if not check_func(length, width, height):
                # print(f"  -> 丢弃: {filename} (L:W:H = {length/width:.1f}:{width/width:.1f}:{height/width:.1f})")
                dropped_count += 1
                continue

            # ==================================
            # 5. 导出合格模型并修复乱码
            # ==================================
            obj_out_path = os.path.join(out_dir, filename)
            mesh.export(obj_out_path)

            with open(obj_out_path, 'r', encoding='utf-8') as f: 
                lines = f.readlines()
            with open(obj_out_path, 'w', encoding='utf-8') as f:
                clean_name = filename.replace('.obj', '')
                for line in lines:
                    if line.startswith('o ') or line.startswith('g '):
                        f.write(f"o {clean_name}\n")
                    else: 
                        f.write(line)

            count += 1
            if count % 20 == 0:
                print(f"  -> 已保留 {count} 个合格模型...")

        except Exception as e:
            # print(f"  -> 读取或处理失败跳过: {filename}，原因: {e}")
            dropped_count += 1

    print(f"\n📊 [{category}] 清洗报告：")
    print(f"  - 输入总数: {len(obj_files)}")
    print(f"  - ✅ 成功保留(原尺寸合格): {count} 个")
    print(f"  - ❌ 几何特征不符剔除: {dropped_count} 个")
    print(f"  - 干净模型保存至: {out_dir}")

print("\n🎉 所有类别清洗完毕！")