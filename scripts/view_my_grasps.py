import argparse
import numpy as np
import trimesh
import trimesh.transformations as tra
from grasp_gen.dataset.eval_utils import load_from_isaac_grasp_format
from grasp_gen.utils.meshcat_utils import create_visualizer, visualize_mesh, visualize_grasp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", type=str, default="/results/my_generated_grasps.yaml")
    parser.add_argument("--mesh_file", type=str, default="/results/tutorial/tutorial_object_dataset/knife_geo_up.obj")
    args = parser.parse_args()

    print("Connecting to Meshcat server...")
    vis = create_visualizer()

    print(f"Loading mesh: {args.mesh_file}")
    obj_mesh = trimesh.load(args.mesh_file)
    # 绝对坐标，不做平移
    visualize_mesh(vis, "scene/object", obj_mesh, color=[169, 169, 169])

    print(f"Loading grasps from: {args.yaml_file}")
    grasps_matrix_list, confidences = load_from_isaac_grasp_format(args.yaml_file)
    print(f"✅ Successfully loaded {len(grasps_matrix_list)} grasps!")

    # 视觉正骨：把 Isaac 的 Z 轴朝向，适配到 Franka 3D 模型的 X 轴朝向
    R_fix = tra.euler_matrix(0, np.pi/2, 0)
    gripper_name = "franka_panda"

    for i, pose_matrix in enumerate(grasps_matrix_list):
        # 仅在显示时旋转 90 度，不改变原数据
        fixed_pose = pose_matrix @ R_fix
        
        visualize_grasp(
            vis,
            name=f"scene/grasps/grasp_{i:03d}",
            transform=fixed_pose,
            color=[0, 255, 0],
            gripper_name=gripper_name,
            linewidth=0.6
        )

    print("\n🎉 Render Complete!")
    print("👉 Open http://127.0.0.1:7000/static/ in your browser.")

if __name__ == "__main__":
    main()
