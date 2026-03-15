import argparse
import numpy as np
import trimesh

from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_grasp,
    visualize_mesh,
)
from grasp_gen.dataset.eval_utils import load_from_isaac_grasp_format, save_to_isaac_grasp_format

def main():
    parser = argparse.ArgumentParser(description="Test GT Grasp Loading, Visualization, and Saving")
    parser.add_argument("--yaml_file", type=str, default="/home/zyp/Desktop/zyp_dataset/knife_filtered_up.yaml")
    parser.add_argument("--mesh_file", type=str, default="/home/zyp/Desktop/zyp_dataset/tutorial/tutorial_object_dataset/knife_geo_up.obj")
    parser.add_argument("--output_file", type=str, default="/home/zyp/Desktop/zyp_dataset/test_passthrough.yaml")
    args = parser.parse_args()

    print("Connecting to Meshcat server...")
    vis = create_visualizer()

    # 1. 加载刀具模型 (不做任何移动)
    print(f"Loading mesh: {args.mesh_file}")
    obj_mesh = trimesh.load(args.mesh_file)
    visualize_mesh(vis, "object_mesh", obj_mesh, color=[169, 169, 169])

    # 2. 调用 GraspGen 自带的函数加载真值数据
    print(f"Loading grasps from: {args.yaml_file}")
    grasps_matrix_list, confidences = load_from_isaac_grasp_format(args.yaml_file)
    print(f"Loaded {len(grasps_matrix_list)} grasps.")

    # 3. 使用 GraspGen 自带的可视化函数进行渲染
    # 你这里可以换成 "robotiq_2f_140" 看是不是夹爪模型不一样导致的
    gripper_name = "franka_panda" 
    for i, grasp in enumerate(grasps_matrix_list):
        # 为了不卡死浏览器，最多只画前 50 个
        if i >= 50: 
            break
        visualize_grasp(
            vis,
            f"grasps_gt/{i:03d}/grasp",
            grasp,
            color=[0, 255, 0], # 画成绿色
            gripper_name=gripper_name,
            linewidth=0.6,
        )

    # 4. 原封不动地存出去，测试 IO 闭环
    print(f"Saving pass-through grasps to: {args.output_file}")
    save_to_isaac_grasp_format(grasps_matrix_list, confidences, args.output_file)
    print("✅ Test finished!")

if __name__ == "__main__":
    main()