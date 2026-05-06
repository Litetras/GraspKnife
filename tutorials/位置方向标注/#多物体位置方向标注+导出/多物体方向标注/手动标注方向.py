import open3d as o3d
import os
import glob
import json
import numpy as np

# 建立轴向输入到 3D 向量的映射表
AXIS_MAP = {
    '+x': [1.0, 0.0, 0.0], '-x': [-1.0, 0.0, 0.0],
    '+y': [0.0, 1.0, 0.0], '-y': [0.0, -1.0, 0.0],
    '+z': [0.0, 0.0, 1.0], '-z': [0.0, 0.0, -1.0]
}

# 物理轴的反向映射 (自动补全用)
OPPOSITE_AXIS = {
    '+x': '-x', '-x': '+x',
    '+y': '-y', '-y': '+y',
    '+z': '-z', '-z': '+z'
}

# 顶会标准：操作者中心坐标系集合 O
VALID_ORIENTATIONS = ['Up', 'Down', 'Left', 'Right', 'Front', 'Back']

# 语义方向的反向映射 (自动补全用)
OPPOSITE_ORI = {
    'Up': 'Down', 'Down': 'Up',
    'Front': 'Back', 'Back': 'Front',
    'Left': 'Right', 'Right': 'Left'
}

def annotate_category_directions(base_dir, output_json):
    print("="*70)
    print("🧭 启动【操作者中心坐标系 (Operator-Centric Frame)】标注器")
    print("📚 严格遵循集合 O = {Up, Down, Left, Right, Front, Back}")
    print("💡 轴向颜色记忆法则：[红=X轴] | [绿=Y轴] | [蓝=Z轴]")
    print("🚀 效率提示：标注 Up 会自动补全 Down，标注 Front 自动补全 Back！")
    print("="*70)

    if os.path.exists(output_json):
        with open(output_json, 'r', encoding='utf-8') as f:
            category_annotations = json.load(f)
        print(f"📦 已加载 {len(category_annotations)} 个类别的历史方向数据。")
    else:
        category_annotations = {}

    folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    for category_folder in folders:
        if category_folder in category_annotations:
            continue

        folder_path = os.path.join(base_dir, category_folder)
        mesh_files = glob.glob(os.path.join(folder_path, '*.obj')) + glob.glob(os.path.join(folder_path, '*.glb'))
        
        if not mesh_files:
            continue
            
        representative_mesh_path = mesh_files[0]
        base_name = os.path.basename(representative_mesh_path)

        print("\n" + "="*70)
        print(f"🏷️  正在标注类别: 【{category_folder}】 (代表模型: {base_name})")
        print("👀 请在弹出的窗口中观察坐标轴。假设【你正握持该工具准备工作】。")
        print("   -> 观察完毕后，按 [Q] 键关闭窗口，在终端进行输入。")
        
        mesh = o3d.io.read_triangle_mesh(representative_mesh_path)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        max_extent = np.max(bbox.get_max_bound() - bbox.get_min_bound())
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max_extent * 0.6, origin=center)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"观察方向: {category_folder}", width=1024, height=768)
        vis.add_geometry(mesh)
        vis.add_geometry(coord_frame)
        vis.run()
        vis.destroy_window()

        directions_for_this_category = {}
        
        while True:
            # 检查还有哪些方向没有标注
            missing_oris = [ori for ori in VALID_ORIENTATIONS if ori not in directions_for_this_category]
            
            if not missing_oris:
                print("🎉 该类别的 6 个标准方向已全部标注完毕！自动进入下一类别。")
                break

            print("\n" + "-"*50)
            print(f"待标注的方向: {missing_oris}")
            print("目前已记录:", list(directions_for_this_category.keys()))
            
            semantic_label = input(f"💬 请选择一个你要标注的语义方向 (如 'Up', 'Front') [输入 'q' 完成当前类别]: ").strip().capitalize()
            
            if semantic_label == 'Q':
                if len(directions_for_this_category) == 0:
                    print("⚠️ 未输入任何方向，暂时跳过该类别。")
                break
                
            if semantic_label not in VALID_ORIENTATIONS:
                print(f"❌ 错误！必须严格使用集合 O 中的词汇: {VALID_ORIENTATIONS}")
                continue

            if semantic_label in directions_for_this_category:
                print(f"⚠️ 方向 '{semantic_label}' 已经标注过了！")
                continue

            axis_input = input(f"👉 哪个物理轴对应操作者的 '{semantic_label}' 方向？(输入 +x, -x, +y, -y, +z, -z): ").strip().lower()
            
            if axis_input not in AXIS_MAP:
                print("❌ 无效的轴向！请输入 +x, -x, +y, -y, +z, 或 -z。")
                continue
                
            # 1. 记录用户输入的方向
            directions_for_this_category[semantic_label] = {
                "axis_str": axis_input,
                "vector": AXIS_MAP[axis_input]
            }
            print(f"✅ 已记录: '{semantic_label}' -> {axis_input}")

            # 2. 自动反推对立方向 (Auto-Inference)
            opp_ori = OPPOSITE_ORI[semantic_label]
            opp_axis = OPPOSITE_AXIS[axis_input]
            
            if opp_ori not in directions_for_this_category:
                directions_for_this_category[opp_ori] = {
                    "axis_str": opp_axis,
                    "vector": AXIS_MAP[opp_axis]
                }
                print(f"🤖 [自动推导]: 侦测到 '{semantic_label}'为 {axis_input}，已自动将 '{opp_ori}' 绑定为 {opp_axis}！")

        # 存入总字典并实时写入 JSON
        category_annotations[category_folder] = directions_for_this_category
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(category_annotations, f, indent=4)
            
        print(f"💾 类别 {category_folder} 配置已保存！")

    print("\n" + "="*70)
    print(f"🏆 所有类别的基准坐标系标注完毕！数据已保存至: {output_json}")

if __name__ == "__main__":
    BASE_DIR = "/home/zyp/Desktop/dataset_obj"  
    OUTPUT_JSON = "category_grasp_directions.json"                  
    annotate_category_directions(BASE_DIR, OUTPUT_JSON)