import yaml
import json
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def convert_isaac_to_graspgen(yaml_file_path, output_dir):
    # 读取 Isaac Sim 生成的 YAML 文件
    with open(yaml_file_path, 'r') as f:
        isaac_data = yaml.safe_load(f)

    # 提取物体信息，假设你要把 .usd 替换成对应的 .obj 用于训练
    object_file_usd = Path(isaac_data.get('object_file', 'object.usd'))
    object_basename = object_file_usd.stem
    object_file_obj = f"{object_basename}.obj"  # GraspGen 通常使用 .obj
    object_scale = isaac_data.get('object_scale', 1.0)

    transforms = []
    object_in_gripper = []

    # 遍历所有抓取姿态
    grasps = isaac_data.get('grasps', {})
    for grasp_id, grasp_data in grasps.items():
        # 1. 提取平移向量 (Translation)
        pos = grasp_data['position']
        
        # 2. 提取四元数 (Quaternion)
        # Isaac Sim 的格式是 w, [x, y, z]
        w = grasp_data['orientation']['w']
        x, y, z = grasp_data['orientation']['xyz']
        
        # scipy Rotation 默认接受的格式是 [x, y, z, w]
        rotation = R.from_quat([x, y, z, w])
        rot_matrix = rotation.as_matrix()

        # 3. 构建 4x4 齐次变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix
        transform_matrix[:3, 3] = pos
        
        transforms.append(transform_matrix.tolist())

        # 4. 提取置信度 (Confidence) 转换为 0 或 1 标签
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

    # 保存文件
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    output_json_path = output_dir_path / f"{object_basename}_grasps.json"
    with open(output_json_path, 'w') as f:
        json.dump(graspgen_dict, f, indent=2)

    print(f"转换成功！")
    print(f"总抓取数: {len(transforms)}")
    print(f"正样本数: {sum(object_in_gripper)}")
    print(f"已保存至: {output_json_path}")

if __name__ == "__main__":
    # 你可以在这里直接修改路径，或者使用 argparse 传参
    YAML_PATH = "/home/zyp/Desktop/zyp_dataset/knife_filtered_down.yaml" # 替换为你的 yaml 文件路径
    OUTPUT_DIR = "/home/zyp/Desktop/zyp_dataset"
    
    convert_isaac_to_graspgen(YAML_PATH, OUTPUT_DIR)