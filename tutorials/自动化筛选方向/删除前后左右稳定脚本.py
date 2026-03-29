import yaml
import os
import glob
import math 

def process_and_split_grasps(yaml_path, out_dir_up, out_dir_down, z_threshold=0.02, y_threshold=0.02, x_limit=0.15, max_tilt_angle=20.0):
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

    for grasp_id, grasp_info in data['grasps'].items():
        # 获取位置信息
        x_pos = grasp_info['position'][0]
        y_pos = grasp_info['position'][1]
        z_pos = grasp_info['position'][2]
        
        # 获取姿态信息 (四元数)
        q = grasp_info['orientation']
        w, qx, qy, qz = q['w'], q['xyz'][0], q['xyz'][1], q['xyz'][2]
        
        # --- 位置过滤 ---
        is_not_side = abs(z_pos) < z_threshold
        is_not_front_back_y = abs(y_pos) > y_threshold
        is_not_front_back_x = abs(x_pos) < x_limit

        # --- 姿态过滤 ---
        # 计算夹爪接近向量在世界坐标系 Y 轴上的投影
        v_y = 2 * (qy * qz - qx * w)
        
        # 限制在 [-1, 1] 防止浮点精度报错
        v_y_clamped = max(-1.0, min(1.0, v_y)) 
        
        # 计算接近方向与竖直 Y 轴的绝对夹角 (0~90度)
        angle_with_vertical = math.degrees(math.acos(abs(v_y_clamped)))
        
        # 判断倾斜角是否在允许范围内 (放宽到 20 度)
        is_vertical = angle_with_vertical < max_tilt_angle

        # --- 终极裁决 ---
        if is_not_side and is_not_front_back_y and is_not_front_back_x and is_vertical:
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
    removed_total = original_count - kept_total
    print(f"Processed {filename}: Up={len(up_grasps)}, Down={len(down_grasps)} (Removed {removed_total} messy grasps)")

def process_all(input_dir, out_dir_up, out_dir_down):
    os.makedirs(out_dir_up, exist_ok=True)
    os.makedirs(out_dir_down, exist_ok=True)

    yaml_files = glob.glob(os.path.join(input_dir, '*.yaml'))
    print(f"Found {len(yaml_files)} YAML files. Starting strict vertical batch processing...\n")
    
    for yaml_file in yaml_files:
        process_and_split_grasps(yaml_file, out_dir_up, out_dir_down, 
                                 z_threshold=0.02, 
                                 y_threshold=0.02, 
                                 x_limit=0.15,
                                 max_tilt_angle=15.0) # 已修改为正负15度
        
    print(f"\nAll done! \nUp grasps saved to: {out_dir_up}\nDown grasps saved to: {out_dir_down}")

if __name__ == "__main__":
    INPUT_DIR = "/home/zyp/GraspDataGen/grasp_guess_data/franka_panda/" 
    OUTPUT_DIR_UP = "/home/zyp/Desktop/knives_only_up/" 
    OUTPUT_DIR_DOWN = "/home/zyp/Desktop/knives_only_down/" 
    
    process_all(INPUT_DIR, OUTPUT_DIR_UP, OUTPUT_DIR_DOWN)