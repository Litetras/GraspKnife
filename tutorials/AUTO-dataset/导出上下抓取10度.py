import yaml
import os
import glob
import math 

def process_and_split_spoons_grasps(yaml_path, out_dir_z, out_dir_neg_y, max_tilt_angle=10.0):
    filename = os.path.basename(yaml_path)
    base_name = filename.replace('.yaml', '')
    
    path_z = os.path.join(out_dir_z, f"{base_name}_z.yaml")
    path_neg_y = os.path.join(out_dir_neg_y, f"{base_name}_neg_y.yaml")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    original_count = len(data.get('grasps', {}))
    if original_count == 0:
        return

    z_grasps = {}
    neg_y_grasps = {}

    for grasp_id, grasp_info in data['grasps'].items():
        # 获取位置信息
        x_pos = grasp_info['position'][0]
        y_pos = grasp_info['position'][1]
        z_pos = grasp_info['position'][2]
        
        # 获取姿态信息 (四元数)
        q = grasp_info['orientation']
        w, qx, qy, qz = q['w'], q['xyz'][0], q['xyz'][1], q['xyz'][2]
        
        # ==========================================
        # 核心：100% 照搬你参考代码中的四元数解析逻辑
        # ==========================================
        # 1. 夹爪接近向量在世界坐标系 Y 轴上的投影 (完全照搬你的原代码)
        v_y = 2 * (qy * qz - qx * w)
        
        # 2. 夹爪接近向量在世界坐标系 Z 轴上的投影 (基于同源数学公式推导)
        v_z = 1 - 2 * (qx**2 + qy**2)
        
        # 限制在 [-1, 1] 防止浮点精度报错
        v_y_clamped = max(-1.0, min(1.0, v_y)) 
        v_z_clamped = max(-1.0, min(1.0, v_z)) 
        
        # 计算接近方向与 Y 轴、Z 轴的绝对夹角 (0~90度)
        angle_with_y = math.degrees(math.acos(abs(v_y_clamped)))
        angle_with_z = math.degrees(math.acos(abs(v_z_clamped)))

        # ==========================================
        # 终极裁决：只看 ±Z 和 -Y (严格 10 度)
        # ==========================================
        
        # 判定 1：±Z 方向 (abs() 已经同时包含了正向和反向)
        if angle_with_z <= max_tilt_angle:
            z_grasps[grasp_id] = grasp_info
            
        # 判定 2：-Y 方向 (角度对齐 Y 轴，且位置处于 Y 轴负半区)
        elif angle_with_y <= max_tilt_angle and y_pos < 0.0:
            neg_y_grasps[grasp_id] = grasp_info

    # 写入文件
    if z_grasps:
        data_z = dict(data) 
        data_z['grasps'] = z_grasps
        with open(path_z, 'w') as f:
            yaml.dump(data_z, f, sort_keys=False)

    if neg_y_grasps:
        data_neg_y = dict(data)
        data_neg_y['grasps'] = neg_y_grasps
        with open(path_neg_y, 'w') as f:
            yaml.dump(data_neg_y, f, sort_keys=False)

    kept_total = len(z_grasps) + len(neg_y_grasps)
    removed_total = original_count - kept_total
    print(f"[{filename}] 保留: ±Z={len(z_grasps)}, -Y={len(neg_y_grasps)} | 剔除: {removed_total}")


def process_all(input_dir, out_dir_z, out_dir_neg_y):
    os.makedirs(out_dir_z, exist_ok=True)
    os.makedirs(out_dir_neg_y, exist_ok=True)

    yaml_files = glob.glob(os.path.join(input_dir, '*.yaml'))
    print(f"找到 {len(yaml_files)} 个抓取文件。开始执行勺子严格 10 度提纯...\n")
    
    for yaml_file in yaml_files:
        # 严格设定为 10 度容差
        process_and_split_spoons_grasps(yaml_file, out_dir_z, out_dir_neg_y, max_tilt_angle=10.0)
        
    print(f"\n全部完成！ \n±Z 抓取存放至: {out_dir_z}\n-Y 抓取存放至: {out_dir_neg_y}")

if __name__ == "__main__":
    INPUT_DIR = "/home/zyp/pan1/objaverse_dataset_5/spoons_grasp/" 
    OUTPUT_DIR_Z = "/home/zyp/pan1/objaverse_dataset_5/spoons_grasps_z_pos/" 
    OUTPUT_DIR_NEG_Y = "/home/zyp/pan1/objaverse_dataset_5/spoons_grasps_neg_y_pos/" 
    
    process_all(INPUT_DIR, OUTPUT_DIR_Z, OUTPUT_DIR_NEG_Y)