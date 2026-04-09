import yaml
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def convert_isaac_to_graspgen(yaml_file_path):
    yaml_path = Path(yaml_file_path)
    
    # 检查文件是否存在
    if not yaml_path.exists():
        print(f"[跳过] 文件不存在: {yaml_path}")
        return None

    # 读取 Isaac Sim 生成的 YAML 文件
    with open(yaml_path, 'r') as f:
        isaac_data = yaml.safe_load(f)

    # 提取物体信息，将 .usd 替换成对应的 .obj
    object_file_usd = Path(isaac_data.get('object_file', 'object.usd'))
    object_basename = object_file_usd.stem
    object_file_obj = f"{object_basename}.obj"  
    object_scale = isaac_data.get('object_scale', 1.0)

    transforms = []
    object_in_gripper = []

    # 遍历所有抓取姿态
    grasps = isaac_data.get('grasps', {})
    for grasp_id, grasp_data in grasps.items():
        pos = grasp_data['position']
        
        # 提取四元数 (Isaac Sim 格式: w, x, y, z)
        w = grasp_data['orientation']['w']
        x, y, z = grasp_data['orientation']['xyz']
        
        # scipy Rotation 默认接受的格式是 [x, y, z, w]
        rotation = R.from_quat([x, y, z, w])
        rot_matrix = rotation.as_matrix()

        # 构建 4x4 齐次变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix
        transform_matrix[:3, 3] = pos
        
        transforms.append(transform_matrix.tolist())

        # 提取置信度 (Confidence) 转换为 0 或 1 标签
        confidence = grasp_data.get('confidence', 1.0)
        is_success = 1 if confidence >= 0.5 else 0
        object_in_gripper.append(is_success)

    # 构建 GraspGen 所需的 JSON 字典结构
    graspgen_dict = {
        "object": {
            "file": object_file_obj,
            "scale": object_scale
        },
        "grasps": {
            "transforms": transforms,
            "object_in_gripper": object_in_gripper
        }
    }

    # 设置输出路径并保存 (直接在原目录下生成 原名_grasps.json)
    output_dir_path = yaml_path.parent
    output_json_path = output_dir_path / f"{yaml_path.stem}_grasps.json"
    
    with open(output_json_path, 'w') as f:
        json.dump(graspgen_dict, f, indent=2)

    print(f"[成功转换] {yaml_path.name} -> {output_json_path.name}")
    return output_json_path

if __name__ == "__main__":
    # 定义基础数据集目录
    BASE_DIR = Path("/home/zyp/Desktop/zyp_dataset7/tutorial/tutorial_grasp_dataset")
    
    print(f"开始扫描目录 {BASE_DIR} 并执行批量转换...")
    
    # 获取目录下所有的 .yaml 文件 
    yaml_files = list(BASE_DIR.glob("*.yaml"))
    
    if not yaml_files:
        print("未找到任何 .yaml 文件，请检查路径。")
    
    # 遍历处理，一行代码搞定
    for target_yaml_path in yaml_files:
        convert_isaac_to_graspgen(target_yaml_path)

    print("\n所有任务完成！")