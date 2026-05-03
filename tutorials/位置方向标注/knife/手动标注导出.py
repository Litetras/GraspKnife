import yaml
import os
import glob
import math
import json

def process_and_split_grasps(yaml_path, boundary_info, out_dirs, z_threshold, y_threshold, x_limit, max_tilt_angle):
    filename = os.path.basename(yaml_path)
    base_name = filename.replace('.yaml', '')
    
    # --- 1. 直接从标注数据中获取完美的边界信息 ---
    if base_name not in boundary_info:
        print(f"[{base_name}] Skipped: 未在标注 JSON 中找到该模型的数据。")
        return
        
    info = boundary_info[base_name]
    major_axis = info["major_axis"]
    boundary_coord = info["boundary_coord"]
    handle_is_positive = info["handle_is_positive"]

    # --- 2. 读取抓取数据 ---
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    original_count = len(data.get('grasps', {}))
    if original_count == 0:
        return

    # 准备四个容器存放分类后的抓取
    categorized_grasps = {
        'up_handle': {}, 'up_blade': {},
        'down_handle': {}, 'down_blade': {}
    }

    for grasp_id, grasp_info in data['grasps'].items():
        pos = grasp_info['position']
        x_pos, y_pos, z_pos = pos[0], pos[1], pos[2]
        
        q = grasp_info['orientation']
        w, qx, qy, qz = q['w'], q['xyz'][0], q['xyz'][1], q['xyz'][2]
        
        # --- 空间位置过滤 ---
        is_not_side = abs(z_pos) < z_threshold
        is_not_front_back_y = abs(y_pos) > y_threshold
        is_not_front_back_x = abs(x_pos) < x_limit

        # --- 接近方向过滤 ---
        v_y = 2 * (qy * qz - qx * w)
        v_y_clamped = max(-1.0, min(1.0, v_y)) 
        angle_with_vertical = math.degrees(math.acos(abs(v_y_clamped)))
        is_vertical = angle_with_vertical < max_tilt_angle

        # 满足基础要求的优质抓取才进行语义分类
        if is_not_side and is_not_front_back_y and is_not_front_back_x and is_vertical:
            
            # 1. 判断全局几何方向 (Up / Down)
            direction = "up" if y_pos > 0.0 else "down"
            
            # 2. 判断功能部件 (Handle / Blade)
            grasp_major_val = pos[major_axis]
            if handle_is_positive:
                # 刀柄在正半区，坐标大于边界即为刀柄
                is_handle = grasp_major_val > boundary_coord
            else:
                # 刀柄在负半区，坐标小于等于边界即为刀柄
                is_handle = grasp_major_val <= boundary_coord
                
            part = "handle" if is_handle else "blade"
            
            # 3. 组合 Key 并存入
            key = f"{direction}_{part}"
            categorized_grasps[key][grasp_id] = grasp_info

    # --- 写入四个独立文件 ---
    kept_total = 0
    for key, grasps in categorized_grasps.items():
        if grasps:
            out_path = os.path.join(out_dirs[key], f"{base_name}_{key}.yaml")
            new_data = dict(data)  # 保留原有 header 等信息
            new_data['grasps'] = grasps
            with open(out_path, 'w') as f:
                yaml.dump(new_data, f, sort_keys=False)
            kept_total += len(grasps)

    removed_total = original_count - kept_total
    print(f"[{base_name}] Extracted: "
          f"U_Hand={len(categorized_grasps['up_handle']):<4} | "
          f"U_Blad={len(categorized_grasps['up_blade']):<4} | "
          f"D_Hand={len(categorized_grasps['down_handle']):<4} | "
          f"D_Blad={len(categorized_grasps['down_blade']):<4} "
          f"(Filtered {removed_total})")


def process_all(input_dir, json_path, output_root, **kwargs):
    # --- 加载 JSON 标注文件 ---
    if not os.path.exists(json_path):
        print(f"Error: 找不到标注文件 {json_path} ! 请检查路径。")
        return
        
    with open(json_path, 'r') as f:
        boundary_info = json.load(f)

    # 创建 4 个语义分类输出文件夹
    categories = ['up_handle', 'up_blade', 'down_handle', 'down_blade']
    out_dirs = {cat: os.path.join(output_root, cat) for cat in categories}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    yaml_files = glob.glob(os.path.join(input_dir, '*.yaml'))
    print(f"Found {len(yaml_files)} YAML files.")
    print(f"Starting Semantic Separation using Manual Annotations...\n" + "="*60)
    
    for yaml_file in yaml_files:
        process_and_split_grasps(
            yaml_path=yaml_file, 
            boundary_info=boundary_info, 
            out_dirs=out_dirs,
            **kwargs
        )
        
    print("="*60 + f"\n[Done] All grasps structurally grounded and separated!")
    print(f"Output saved to: {output_root}")


if __name__ == "__main__":
    # ------------------ 配置区 ------------------
    # 1. 原始全局抓取 YAML 文件目录
    INPUT_DIR = "/home/zyp/GraspDataGen/grasp_guess_data/franka_panda/" 
    
    # 2. 刚才生成的标注文件 JSON 路径 (请确保路径正确，如果在同级目录直接写文件名即可)
    JSON_PATH = "knife_boundaries.json" 
    
    # 3. 最终输出的根目录（会自动创建 4 个子文件夹）
    OUTPUT_ROOT = "/home/zyp/Desktop/knives_grounded_grasps/" 
    
    # 执行批处理
    process_all(
        input_dir=INPUT_DIR, 
        json_path=JSON_PATH, 
        output_root=OUTPUT_ROOT,
        z_threshold=0.02, 
        y_threshold=0.02, 
        x_limit=0.15,
        max_tilt_angle=10.0  # 沿用你收紧后的 10度 限制
    )