import yaml
import os
import glob
import math 

def process_and_split_hammer_grasps(yaml_path, out_dir_up, out_dir_down, z_threshold=0.04, y_threshold=0.005, x_min=0.02, x_max=0.35, max_tilt_angle=12.0):
    filename = os.path.basename(yaml_path)
    base_name = filename.replace('.yaml', '')
    
    path_up = os.path.join(out_dir_up, f"{base_name}_up.yaml")
    path_down = os.path.join(out_dir_down, f"{base_name}_down.yaml")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    original_count = len(data.get('grasps', {}))
    if original_count == 0:
        return

    up_grasps = {}
    down_grasps = {}

    # --- 诊断统计器 ---
    fail_z = 0
    fail_x = 0
    fail_y = 0
    fail_angle = 0

    for grasp_id, grasp_info in data['grasps'].items():
        x_pos = grasp_info['position'][0]
        y_pos = grasp_info['position'][1]
        z_pos = grasp_info['position'][2]
        
        q = grasp_info['orientation']
        w, qx, qy, qz = q['w'], q['xyz'][0], q['xyz'][1], q['xyz'][2]
        
        # 1. Z轴居中判断 (放宽到 4cm，适应粗把手和弯把手)
        is_centered_z = abs(z_pos) < z_threshold
        
        # 2. X轴手柄区域 (x_min 放宽到 0.02 靠近锤头一点，x_max 放宽到 0.35)
        is_on_handle_x = x_min < x_pos < x_max
        
        # 3. Y轴上下偏移 (放宽到 0.005，0.5厘米，适应极细的把手)
        is_offset_y = abs(y_pos) > y_threshold

        # 4. 角度判断
        v_y = 2 * (qy * qz - qx * w)
        v_y_clamped = max(-1.0, min(1.0, v_y)) 
        angle_with_vertical = math.degrees(math.acos(abs(v_y_clamped)))
        is_vertical = angle_with_vertical < max_tilt_angle

        # --- 记录死亡原因并裁决 ---
        if not is_centered_z:
            fail_z += 1
        elif not is_on_handle_x:
            fail_x += 1
        elif not is_offset_y:
            fail_y += 1
        elif not is_vertical:
            fail_angle += 1
        else:
            # 存活下来的
            if y_pos > 0.0:
                up_grasps[grasp_id] = grasp_info
            else:
                down_grasps[grasp_id] = grasp_info

    # 写入文件
    if up_grasps:
        data_up = dict(data) 
        data_up['grasps'] = up_grasps
        with open(path_up, 'w') as f:
            yaml.dump(data_up, f, sort_keys=False)

    if down_grasps:
        data_down = dict(data)
        data_down['grasps'] = down_grasps
        with open(path_down, 'w') as f:
            yaml.dump(data_down, f, sort_keys=False)

    kept_total = len(up_grasps) + len(down_grasps)
    
    # 打印诊断报告
    print(f"[{base_name}] 保留: {kept_total} (Up:{len(up_grasps)}, Down:{len(down_grasps)}) | "
          f"死因排查 -> 抓偏(Z):{fail_z}, 抓头(X):{fail_x}, 贴脸(Y):{fail_y}, 角度歪(Angle):{fail_angle}")

def process_all(input_dir, out_dir_up, out_dir_down):
    os.makedirs(out_dir_up, exist_ok=True)
    os.makedirs(out_dir_down, exist_ok=True)

    yaml_files = glob.glob(os.path.join(input_dir, '*.yaml'))
    print(f"找到 {len(yaml_files)} 个YAML文件。开启诊断模式...\n")
    
    for yaml_file in yaml_files:
        process_and_split_hammer_grasps(yaml_file, out_dir_up, out_dir_down)
        
    print(f"\n全部完成！")

if __name__ == "__main__":
    INPUT_DIR = "/home/zyp/GraspDataGen/grasp_guess_data/franka_panda/" 
    OUTPUT_DIR_UP = "/home/zyp/Desktop/hammers_only_up/" 
    OUTPUT_DIR_DOWN = "/home/zyp/Desktop/hammers_only_down/" 
    
    process_all(INPUT_DIR, OUTPUT_DIR_UP, OUTPUT_DIR_DOWN)