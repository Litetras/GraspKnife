import yaml
import os
import glob
import math 
import random  # 新增 random 库用于均匀抽样

def process_and_split_mugs_grasps(yaml_path, out_dir_x, out_dir_z, 
                                  z_mouth_angle=10.0,     # 杯口抓取：总角度容差保持 10度
                                  x_handle_yaw=15.0,      # 把手抓取：左右防撞容差 15度
                                  x_handle_pitch=45.0,    # 把手抓取：上下倾斜容差 放宽到 45度
                                  x_handle_roll=15.0,     # 把手抓取：手指水平容差 15度
                                  max_z_grasps=200):      # ★ 新增：杯口抓取最大保留数量
    
    filename = os.path.basename(yaml_path)
    base_name = filename.replace('.yaml', '')
    
    path_x = os.path.join(out_dir_x, f"{base_name}_x.yaml")
    path_z = os.path.join(out_dir_z, f"{base_name}_z.yaml")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    original_count = len(data.get('grasps', {}))
    if original_count == 0:
        return

    x_grasps = {}
    z_grasps = {}

    for grasp_id, grasp_info in data['grasps'].items():
        x_pos = grasp_info['position'][0]
        y_pos = grasp_info['position'][1]
        z_pos = grasp_info['position'][2]
        
        q = grasp_info['orientation']
        w, qx, qy, qz = q['w'], q['xyz'][0], q['xyz'][1], q['xyz'][2]
        
        # 1. 提取夹爪局部坐标轴
        local_x = [1.0 - 2.0*(qy**2 + qz**2), 2.0*(qx*qy + qz*w),       2.0*(qx*qz - qy*w)]
        local_y = [2.0*(qx*qy - qz*w),        1.0 - 2.0*(qx**2 + qz**2), 2.0*(qy*qz + qx*w)]
        local_z = [2.0*(qx*qz + qy*w),        2.0*(qy*qz - qx*w),       1.0 - 2.0*(qx**2 + qy**2)]

        # 2. 探测真实接近轴
        dist = math.sqrt(x_pos**2 + y_pos**2 + z_pos**2)
        if dist < 1e-6: dist = 1e-6
        dir_to_center = [-x_pos/dist, -y_pos/dist, -z_pos/dist]

        dot_x = abs(sum(a*b for a, b in zip(local_x, dir_to_center)))
        dot_y = abs(sum(a*b for a, b in zip(local_y, dir_to_center)))
        dot_z = abs(sum(a*b for a, b in zip(local_z, dir_to_center)))

        max_dot = max(dot_x, dot_y, dot_z)
        if max_dot == dot_x:
            approach_vec = local_x
        elif max_dot == dot_y:
            approach_vec = local_y
        else:
            approach_vec = local_z

        # ==========================================
        # ★ 判定 A：杯口抓取逻辑 (保持原样，严格 10 度)
        # ==========================================
        v_z_clamped = max(-1.0, min(1.0, approach_vec[2])) 
        angle_with_z = math.degrees(math.acos(abs(v_z_clamped)))
        is_z_aligned = angle_with_z <= z_mouth_angle
        is_z_region = (z_pos < -0.02) 

        # ==========================================
        # ★ 判定 B：把手抓取逻辑 (解耦 XYZ，释放上下角度)
        # ==========================================
        # 1. 左右偏差 (Yaw)：接近向量在 Y 轴的投影大小
        yaw_deviation = math.degrees(math.asin(min(1.0, abs(approach_vec[1]))))
        # 2. 上下偏差 (Pitch)：接近向量在 Z 轴的投影大小
        pitch_deviation = math.degrees(math.asin(min(1.0, abs(approach_vec[2]))))
        
        # 3. 手指水平度 (Roll)：局部 Y 轴 (手指开合轴) 偏离水平面的角度
        finger_tilt_angle = math.degrees(math.asin(min(1.0, abs(local_y[2]))))
        
        # 组装把手的姿态判定
        is_x_aligned = (yaw_deviation <= x_handle_yaw) and \
                       (pitch_deviation <= x_handle_pitch) and \
                       (finger_tilt_angle <= x_handle_roll)

        # 空间位置锁
        is_x_region = (x_pos > 0.0) and (abs(z_pos) < 0.08)

        # ==========================================
        # 终极裁决
        # ==========================================
        if is_x_aligned and is_x_region:
            x_grasps[grasp_id] = grasp_info
            
        elif is_z_aligned and is_z_region:
            z_grasps[grasp_id] = grasp_info

    # ==========================================
    # ★ 核心新增：随机截断 Z 方向抓取数量 ★
    # ==========================================
    if len(z_grasps) > max_z_grasps:
        # 随机抽取 200 个键，保证空间分布的多样性
        sampled_keys = random.sample(list(z_grasps.keys()), max_z_grasps)
        # 重新构建只包含这 200 个的字典
        z_grasps = {k: z_grasps[k] for k in sampled_keys}

    # 写入文件
    if x_grasps:
        data_x = dict(data) 
        data_x['grasps'] = x_grasps
        with open(path_x, 'w') as f:
            yaml.dump(data_x, f, sort_keys=False)

    if z_grasps:
        data_z = dict(data)
        data_z['grasps'] = z_grasps
        with open(path_z, 'w') as f:
            yaml.dump(data_z, f, sort_keys=False)

    kept_total = len(x_grasps) + len(z_grasps)
    removed_total = original_count - kept_total
    print(f"[{filename}] 保留: 把手(+X)={len(x_grasps)}, 杯口(+Z)={len(z_grasps)} | 剔除: {removed_total}")


def process_all(input_dir, out_dir_x, out_dir_z):
    os.makedirs(out_dir_x, exist_ok=True)
    os.makedirs(out_dir_z, exist_ok=True)

    yaml_files = glob.glob(os.path.join(input_dir, '*.yaml'))
    print(f"找到 {len(yaml_files)} 个抓取文件。开始执行解耦倾角提纯与随机采样截断...\n")
    
    for yaml_file in yaml_files:
        process_and_split_mugs_grasps(yaml_file, out_dir_x, out_dir_z)
        
    print(f"\n全部完成！ \n把手抓取 (+X) 存放至: {out_dir_x}\n杯口抓取 (+Z) 存放至: {out_dir_z}")

if __name__ == "__main__":
    INPUT_DIR = "/home/zyp/pan1/objaverse_dataset_5/mugs_grasp/" 
    OUTPUT_DIR_X = "/home/zyp/pan1/objaverse_dataset_5/mugs_grasps_x_pos/" 
    OUTPUT_DIR_Z = "/home/zyp/pan1/objaverse_dataset_5/mugs_grasps_z_pos/" 
    
    process_all(INPUT_DIR, OUTPUT_DIR_X, OUTPUT_DIR_Z)