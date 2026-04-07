import yaml
import os
import glob
import math
import json

def process_and_split_hammer_grasps(yaml_path, boundary_info, out_dirs, z_threshold=0.04, y_threshold=0.005, max_tilt_angle=12.0):
    filename = os.path.basename(yaml_path)
    base_name = filename.replace('.yaml', '')
    
    # --- 1. 从 JSON 获取完美的边界信息 ---
    if base_name not in boundary_info:
        print(f"[{base_name}] Skipped: 未在标注 JSON 中找到该锤子的数据。")
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

    # 准备四个容器存放分类后的抓取 (包含锤柄和锤头)
    categorized_grasps = {
        'up_handle': {}, 'up_head': {},
        'down_handle': {}, 'down_head': {}
    }

    # --- 诊断统计器 (移除了 fail_x，因为 X 轴现在用于分类而不是淘汰) ---
    fail_z = 0
    fail_y = 0
    fail_angle = 0

    for grasp_id, grasp_info in data['grasps'].items():
        pos = grasp_info['position']
        x_pos, y_pos, z_pos = pos[0], pos[1], pos[2]
        
        q = grasp_info['orientation']
        w, qx, qy, qz = q['w'], q['xyz'][0], q['xyz'][1], q['xyz'][2]
        
        # 1. Z轴居中判断 (放宽到 4cm，适应粗把手和弯把手)
        is_centered_z = abs(z_pos) < z_threshold
        
        # 2. Y轴上下偏移 (放宽到 0.005，0.5厘米，适应极细的把手)
        is_offset_y = abs(y_pos) > y_threshold

        # 3. 角度判断
        v_y = 2 * (qy * qz - qx * w)
        v_y_clamped = max(-1.0, min(1.0, v_y)) 
        angle_with_vertical = math.degrees(math.acos(abs(v_y_clamped)))
        is_vertical = angle_with_vertical < max_tilt_angle

        # --- 记录死亡原因并裁决 ---
        if not is_centered_z:
            fail_z += 1
        elif not is_offset_y:
            fail_y += 1
        elif not is_vertical:
            fail_angle += 1
        else:
            # 存活下来的抓取，开始进行语义分类
            
            # 判断全局几何方向 (Up / Down)
            direction = "up" if y_pos > 0.0 else "down"
            
            # 利用标注判断功能部件 (Handle / Head)
            grasp_major_val = pos[major_axis]
            if handle_is_positive:
                is_handle = grasp_major_val > boundary_coord
            else:
                is_handle = grasp_major_val <= boundary_coord
                
            part = "handle" if is_handle else "head"
            
            # 组合 Key 并存入
            key = f"{direction}_{part}"
            categorized_grasps[key][grasp_id] = grasp_info

    # --- 写入四个独立文件 ---
    kept_total = 0
    for key, grasps in categorized_grasps.items():
        if grasps:
            out_path = os.path.join(out_dirs[key], f"{base_name}_{key}.yaml")
            new_data = dict(data) 
            new_data['grasps'] = grasps
            with open(out_path, 'w') as f:
                yaml.dump(new_data, f, sort_keys=False)
            kept_total += len(grasps)
            
    # 打印诊断报告 (同时显示 4 个分类的数量和被过滤掉的数量)
    print(f"[{base_name}] 保留: {kept_total} "
          f"(U_Hand:{len(categorized_grasps['up_handle'])}, U_Head:{len(categorized_grasps['up_head'])}, "
          f"D_Hand:{len(categorized_grasps['down_handle'])}, D_Head:{len(categorized_grasps['down_head'])}) | "
          f"死因排查 -> 抓偏(Z):{fail_z}, 贴脸(Y):{fail_y}, 角度歪(Angle):{fail_angle}")

def process_all(input_dir, json_path, output_root):
    # --- 加载 JSON 标注文件 ---
    if not os.path.exists(json_path):
        print(f"Error: 找不到标注文件 {json_path} ! 请先运行标注脚本。")
        return
        
    with open(json_path, 'r') as f:
        boundary_info = json.load(f)

    # 创建 4 个语义分类输出文件夹
    categories = ['up_handle', 'up_head', 'down_handle', 'down_head']
    out_dirs = {cat: os.path.join(output_root, cat) for cat in categories}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    yaml_files = glob.glob(os.path.join(input_dir, '*.yaml'))
    print(f"找到 {len(yaml_files)} 个YAML文件。开启精准诊断分类模式...\n" + "="*80)
    
    for yaml_file in yaml_files:
        process_and_split_hammer_grasps(yaml_file, boundary_info, out_dirs)
        
    print("="*80 + f"\n[Done] 全部完成！输出结果在: {output_root}")

if __name__ == "__main__":
    # 1. 原始锤子抓取 YAML 文件目录
    INPUT_DIR = "/home/zyp/GraspDataGen/grasp_guess_data/franka_panda/" 
    
    # 2. 刚刚生成的锤子标注 JSON 路径
    JSON_PATH = "hammer_boundaries.json" 
    
    # 3. 最终输出的根目录（会自动在其下创建 up_handle, up_head 等 4 个子文件夹）
    OUTPUT_ROOT = "/home/zyp/Desktop/hammers_grounded_grasps/" 
    
    process_all(INPUT_DIR, JSON_PATH, OUTPUT_ROOT)