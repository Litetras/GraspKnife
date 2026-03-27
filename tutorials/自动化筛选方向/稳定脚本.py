import yaml
import os
import glob

def process_and_split_grasps(yaml_path, out_dir_up, out_dir_down, z_threshold=0.02):
    filename = os.path.basename(yaml_path)
    # 去掉 .yaml 后缀，方便后面拼接 _up 和 _down
    base_name = filename.replace('.yaml', '')
    
    # 构建输出路径
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
        y_pos = grasp_info['position'][1]
        z_pos = grasp_info['position'][2]
        
        # 1. 过滤掉左右两侧的抓取
        if abs(z_pos) < z_threshold:
            # 2. 区分上方和下方
            if y_pos > 0.0:
                up_grasps[grasp_id] = grasp_info
            else:
                down_grasps[grasp_id] = grasp_info

    # 为了保留原 YAML 里的头部信息 (如 object_file, gripper_file 等)
    # 我们复制一份原始数据，只替换 grasps 字典
    
    # 保存上方抓取
    data_up = dict(data) 
    data_up['grasps'] = up_grasps
    with open(path_up, 'w') as f:
        yaml.dump(data_up, f, sort_keys=False)

    # 保存下方抓取
    data_down = dict(data)
    data_down['grasps'] = down_grasps
    with open(path_down, 'w') as f:
        yaml.dump(data_down, f, sort_keys=False)

    print(f"Processed {filename}: Up={len(up_grasps)}, Down={len(down_grasps)} (Removed {original_count - len(up_grasps) - len(down_grasps)} sides)")

def process_all(input_dir, out_dir_up, out_dir_down):
    # 如果输出文件夹不存在，自动创建它们
    os.makedirs(out_dir_up, exist_ok=True)
    os.makedirs(out_dir_down, exist_ok=True)

    yaml_files = glob.glob(os.path.join(input_dir, '*.yaml'))
    print(f"Found {len(yaml_files)} YAML files. Starting full batch processing...\n")
    
    for yaml_file in yaml_files:
        process_and_split_grasps(yaml_file, out_dir_up, out_dir_down)
        
    print(f"\nAll done! \nUp grasps saved to: {out_dir_up}\nDown grasps saved to: {out_dir_down}")

if __name__ == "__main__":
    # 输入文件夹路径
    INPUT_DIR = "/home/zyp/GraspDataGen/grasp_guess_data/franka_panda/" 
    
    # 输出文件夹路径（分别存放上方和下方的抓取）
    OUTPUT_DIR_UP = "/home/zyp/Desktop/knives_only_up/" 
    OUTPUT_DIR_DOWN = "/home/zyp/Desktop/knives_only_down/" 
    
    # 直接执行全量处理
    process_all(INPUT_DIR, OUTPUT_DIR_UP, OUTPUT_DIR_DOWN)