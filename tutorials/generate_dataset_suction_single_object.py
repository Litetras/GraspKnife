# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
教程：生成单物体吸盘抓取训练数据集

本脚本演示了如何使用吸盘抓取器为单个物体生成用于训练生成器和判别器模型的数据集。
该数据集遵循 GraspGen 约定，可用于训练神经网络以预测抓取姿态。

用法示例:
    python generate_dataset_suction_single_object.py \
        --object_path /path/to/object.obj \
        --output_dir /results \
        --num_grasps 2000
"""

import os
import sys
import json
import argparse
import numpy as np
import trimesh
import trimesh.transformations as tra
from pathlib import Path

# 将父目录添加到路径中以导入 grasp_gen 模块
sys.path.append(str(Path(__file__).parent.parent))

from grasp_gen.dataset.suction import SuctionCupArray


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="生成单物体吸盘抓取训练数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="物体网格文件的路径 (obj, stl, 或 ply 格式)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/results/tutorial",
        help="保存教程数据集的目录",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=2000,
        help="要生成的抓取总数（包含正样本和负样本）",
    )
    parser.add_argument(
        "--object_scale",
        type=float,
        default=1.0,
        help="应用于物体网格的缩放因子",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default="config/grippers/single_suction_cup_30mm.yaml",
        help="抓取器配置文件的路径",
    )
    parser.add_argument(
        "--num_disturbances",
        type=int,
        default=10,
        help="用于评估的随机扰动采样数（测试抓取稳定性）",
    )
    parser.add_argument(
        "--qp_solver",
        type=str,
        default="clarabel",
        choices=("clarabel", "cvxopt", "daqp", "ecos", "osqp", "scs"),
        help="用于力螺旋（wrench）阻力评估的二次规划求解器",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="随机种子，用于结果复现"
    )
    parser.add_argument(
        "--no_visualization", action="store_true", help="禁用可视化（不启动 meshcat）"
    )

    return parser.parse_args()


def setup_directories(output_dir, object_basename):
    """创建教程数据集所需的目录结构。"""
    # 创建教程物体数据集目录 (存放 .obj 等模型文件)
    tutorial_object_dir = Path(output_dir) / "tutorial_object_dataset"
    tutorial_object_dir.mkdir(parents=True, exist_ok=True)

    # 创建教程抓取数据集目录 (存放 .json 等标签文件)
    tutorial_grasp_dir = Path(output_dir) / "tutorial_grasp_dataset"
    tutorial_grasp_dir.mkdir(parents=True, exist_ok=True)

    return tutorial_object_dir, tutorial_grasp_dir


def copy_object_to_dataset(
    object_path, tutorial_object_dir, object_basename, scale=1.0
):
    """将物体网格复制并缩放到教程数据集目录。"""
    # 加载物体网格
    obj_mesh = trimesh.load(object_path)

    # 如果需要，应用缩放
    if scale != 1.0:
        obj_mesh.apply_scale(scale)

    # 保存到教程物体数据集目录
    output_object_path = tutorial_object_dir / f"{object_basename}.obj"
    obj_mesh.export(str(output_object_path))

    print(f"物体已保存至: {output_object_path}")
    return str(output_object_path), obj_mesh


def generate_grasps(obj_mesh, suction_gripper, num_grasps, num_disturbances, qp_solver):
    """生成并评估物体的抓取姿态。"""
    print(f"正在生成 {num_grasps} 个抓取...")

    # 1. 采样抓取
    # 在物体表面随机采样点，并计算接近向量和初步的抓取变换矩阵
    points_on_surface, approach_vectors, grasp_transforms = (
        suction_gripper.sample_grasps(obj_mesh=obj_mesh, num_grasps=num_grasps)
    )

    # 2. 评估抓取
    print("正在评估抓取...")
    # 计算物理属性：是否密封(sealed)、是否成功(success)、是否碰撞(in_collision)等
    points, approach_vectors, contact_transforms, sealed, success, in_collision = (
        suction_gripper.evaluate_grasps(
            obj_mesh=obj_mesh,
            points_on_surface=points_on_surface,
            approach_vectors=approach_vectors,
            grasp_transforms=grasp_transforms,
            num_disturbances=num_disturbances,
            qp_solver=qp_solver,
            tqdm_disable=False, # 显示进度条
        )
    )

    return points, approach_vectors, contact_transforms, sealed, success, in_collision


def create_grasp_dataset_json(
    object_file,
    object_scale,
    contact_transforms,
    sealed,
    success,
    in_collision,
    tutorial_grasp_dir,
    object_basename,
    obj_center_mass,
):
    """按照 GraspGen 的约定创建抓取数据集 JSON 文件。"""

    # 断言：确保 obj_center_mass 是一个 3D 向量
    assert (
        len(obj_center_mass) == 3
    ), f"物体质心长度应为 3，实际为 {len(obj_center_mass)}"

    # 确定成功的抓取 (需要密封、高成功率且无碰撞)
    # 我们使用成功率阈值来混合正样本和负样本
    success_threshold = 0.5  # 成功率 > 50% 的抓取被视为正样本

    # 创建 object_in_gripper 掩码 (标记哪些抓取是有效的)
    # reduce 用于对列表中的数组进行逻辑与操作
    object_in_gripper = np.logical_and.reduce(
        [
            sealed,  # 必须形成真空密封
            # ~in_collision  # 必须没有发生非预期的碰撞; TODO - 检查碰撞网格是否指定正确
        ]
    )

    # 将抓取变换转换回原始物体坐标系 (加回质心偏移)
    # 因为物理评估通常在物体质心为原点时进行，但保存的数据需相对于原始物体位置
    output_transforms = np.copy(contact_transforms)
    output_transforms[:, :3, 3] += obj_center_mass

    # 创建数据集字典结构
    dataset_dict = {
        "object": {
            "file": object_file,  # 教程数据集中的物体相对路径
            "scale": object_scale,
        },
        "grasps": {
            "transforms": output_transforms.tolist(), # 抓取位姿矩阵列表
            "object_in_gripper": object_in_gripper.tolist(), # 标签：0或1 (是否成功抓取)
        },
    }

    # 保存为 JSON 文件
    output_json_path = tutorial_grasp_dir / f"{object_basename}_grasps.json"
    with open(output_json_path, "w") as f:
        json.dump(dataset_dict, f, indent=2)

    print(f"抓取数据集已保存至: {output_json_path}")
    print(f"抓取总数: {len(object_in_gripper)}")
    print(f"正样本抓取数: {sum(object_in_gripper)}")
    print(f"负样本抓取数: {sum(~object_in_gripper)}")

    return str(output_json_path)


def create_splits_file(tutorial_object_dir, tutorial_grasp_dir, object_basename):
    """为教程数据集创建 train.txt 和 valid.txt 分割文件。"""

    # 创建 train.txt (将该物体加入训练集)
    train_txt_path = tutorial_object_dir / "train.txt"
    with open(train_txt_path, "w") as f:
        f.write(f"{object_basename}.obj\n")

    # 创建 valid.txt (将同一个物体加入验证集用于演示)
    valid_txt_path = tutorial_object_dir / "valid.txt"
    with open(valid_txt_path, "w") as f:
        f.write(f"{object_basename}.obj\n")

    print(f"数据集分割文件已创建:")
    print(f"  训练集: {train_txt_path}")
    print(f"  验证集: {valid_txt_path}")


def main():
    """主函数：生成教程数据集。"""
    args = parse_args()

    # 设置随机种子以保证可复现性
    np.random.seed(args.random_seed)

    # 验证输入文件是否存在
    if not os.path.exists(args.object_path):
        raise FileNotFoundError(f"找不到物体文件: {args.object_path}")

    if not os.path.exists(args.gripper_config):
        raise FileNotFoundError(f"找不到抓取器配置: {args.gripper_config}")

    # 获取物体基本名称 (不带扩展名)
    object_basename = Path(args.object_path).stem

    print(f"正在为物体生成数据集: {object_basename}")
    print(f"物体路径: {args.object_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"抓取数量: {args.num_grasps}")
    print(f"物体缩放: {args.object_scale}")

    # 1. 设置目录结构
    tutorial_object_dir, tutorial_grasp_dir = setup_directories(
        args.output_dir, object_basename
    )
    # 2. 复制并处理物体网格
    object_file, obj_mesh = copy_object_to_dataset(
        args.object_path, tutorial_object_dir, object_basename, args.object_scale
    )

    # 使 object_file 成为相对于 tutorial_object_dir 的路径
    object_file = f"{object_basename}.obj"

    # 3. 将物体质心移动到原点
    # 这是物理模拟的关键步骤，许多计算假设物体位于原点
    obj_center_mass = obj_mesh.center_mass
    obj_mesh.apply_translation(-obj_center_mass)
    print(f"物体已居中至原点。原始质心位置: {obj_center_mass}")

    # 4. 初始化吸盘抓取器
    print(f"正在加载抓取器配置: {args.gripper_config}")
    suction_gripper = SuctionCupArray.from_file(fname=args.gripper_config)

    # 5. 生成抓取数据
    points, approach_vectors, contact_transforms, sealed, success, in_collision = (
        generate_grasps(
            obj_mesh=obj_mesh,
            suction_gripper=suction_gripper,
            num_grasps=args.num_grasps,
            num_disturbances=args.num_disturbances,
            qp_solver=args.qp_solver,
        )
    )

    # 6. 创建抓取数据集 JSON (包含标签和位姿)
    grasp_json_path = create_grasp_dataset_json(
        object_file=object_file,
        object_scale=args.object_scale,
        contact_transforms=contact_transforms,
        sealed=sealed,
        success=success,
        in_collision=in_collision,
        tutorial_grasp_dir=tutorial_grasp_dir,
        object_basename=object_basename,
        obj_center_mass=obj_center_mass,
    )

    # 7. 创建分割文件 (定义训练集/验证集列表)
    create_splits_file(tutorial_object_dir, tutorial_grasp_dir, object_basename)

    # 8. 可选的可视化
    if not args.no_visualization:
        print("\n正在可视化结果...")
        from grasp_gen.utils.meshcat_utils import (
            create_visualizer,
            visualize_mesh,
            visualize_pointcloud,
            visualize_grasp,
        )

        vis = create_visualizer()

        # 可视化物体
        visualize_mesh(vis, "object", obj_mesh, color=[169, 169, 169])

        # 可视化抓取点 (根据成功率着色)
        from grasp_gen.dataset.suction import colorize_for_meshcat

        grasp_colors = colorize_for_meshcat(success)
        visualize_pointcloud(vis, "grasp_points", points, grasp_colors, size=0.005)

        # 可视化前 10 个最佳抓取
        # 综合标准 = 密封 * 成功率
        combined_criteria = sealed * success
        top_indices = np.argsort(combined_criteria)[-10:][::-1]

        for i, idx in enumerate(top_indices):
            if combined_criteria[idx] > 0:
                # 最佳抓取为红色，其余为橙色
                color = [255, 0, 0] if i == 0 else [255, 165, 0]
                visualize_grasp(
                    vis,
                    f"top_grasps/grasp_{i:02d}",
                    contact_transforms[idx],
                    color=color,
                    gripper_name="single_suction_cup_30mm",
                )

        print("可视化完成。请检查 meshcat 浏览器窗口。")
        input("按 Enter 键继续...")

    print(f"\n教程数据集生成完成！")
    print(f"物体数据集目录: {tutorial_object_dir}")
    print(f"抓取数据集目录: {tutorial_grasp_dir}")
    print(f"准备开始训练生成器和判别器模型！")


if __name__ == "__main__":
    main()





# # Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# #
# # NVIDIA CORPORATION and its licensors retain all intellectual property
# # and proprietary rights in and to this software, related documentation
# # and any modifications thereto. Any use, reproduction, disclosure or
# # distribution of this software and related documentation without an express
# # license agreement from NVIDIA CORPORATION is strictly prohibited.

# """
# Tutorial: Generate Dataset for Single Object Suction Grasp Training

# This script demonstrates how to generate a dataset for training a generator and discriminator
# model for a single object using a suction cup gripper. The dataset follows the GraspGen
# convention and can be used for training neural networks to predict grasp poses.

# Usage:
#     python generate_dataset_suction_single_object.py \
#         --object_path /path/to/object.obj \
#         --output_dir /results \
#         --num_grasps 2000
# """

# import os
# import sys
# import json
# import argparse
# import numpy as np
# import trimesh
# import trimesh.transformations as tra
# from pathlib import Path

# # Add the parent directory to the path to import grasp_gen modules
# sys.path.append(str(Path(__file__).parent.parent))

# from grasp_gen.dataset.suction import SuctionCupArray


# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Generate dataset for single object suction grasp training",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument(
#         "--object_path",
#         type=str,
#         required=True,
#         help="Path to the object mesh file (obj, stl, or ply)",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="/results/tutorial",
#         help="Directory to save the tutorial datasets",
#     )
#     parser.add_argument(
#         "--num_grasps",
#         type=int,
#         default=2000,
#         help="Total number of grasps to generate (positive and negative)",
#     )
#     parser.add_argument(
#         "--object_scale",
#         type=float,
#         default=1.0,
#         help="Scale factor to apply to the object mesh",
#     )
#     parser.add_argument(
#         "--gripper_config",
#         type=str,
#         default="config/grippers/single_suction_cup_30mm.yaml",
#         help="Path to gripper configuration file",
#     )
#     parser.add_argument(
#         "--num_disturbances",
#         type=int,
#         default=10,
#         help="Number of random disturbance samples for evaluation",
#     )
#     parser.add_argument(
#         "--qp_solver",
#         type=str,
#         default="clarabel",
#         choices=("clarabel", "cvxopt", "daqp", "ecos", "osqp", "scs"),
#         help="QP solver to use for wrench resistance evaluation",
#     )
#     parser.add_argument(
#         "--random_seed", type=int, default=42, help="Random seed for reproducibility"
#     )
#     parser.add_argument(
#         "--no_visualization", action="store_true", help="Disable visualization"
#     )

#     return parser.parse_args()


# def setup_directories(output_dir, object_basename):
#     """Create the required directory structure for the tutorial datasets."""
#     # Create tutorial object dataset directory
#     tutorial_object_dir = Path(output_dir) / "tutorial_object_dataset"
#     tutorial_object_dir.mkdir(parents=True, exist_ok=True)

#     # Create tutorial grasp dataset directory
#     tutorial_grasp_dir = Path(output_dir) / "tutorial_grasp_dataset"
#     tutorial_grasp_dir.mkdir(parents=True, exist_ok=True)

#     return tutorial_object_dir, tutorial_grasp_dir


# def copy_object_to_dataset(
#     object_path, tutorial_object_dir, object_basename, scale=1.0
# ):
#     """Copy and scale the object mesh to the tutorial dataset directory."""
#     # Load the object mesh
#     obj_mesh = trimesh.load(object_path)

#     # Apply scale if needed
#     if scale != 1.0:
#         obj_mesh.apply_scale(scale)

#     # Save to tutorial object dataset
#     output_object_path = tutorial_object_dir / f"{object_basename}.obj"
#     obj_mesh.export(str(output_object_path))

#     print(f"Object saved to: {output_object_path}")
#     return str(output_object_path), obj_mesh


# def generate_grasps(obj_mesh, suction_gripper, num_grasps, num_disturbances, qp_solver):
#     """Generate and evaluate grasps for the object."""
#     print(f"Generating {num_grasps} grasps...")

#     # Sample grasps
#     points_on_surface, approach_vectors, grasp_transforms = (
#         suction_gripper.sample_grasps(obj_mesh=obj_mesh, num_grasps=num_grasps)
#     )

#     # Evaluate grasps
#     print("Evaluating grasps...")
#     points, approach_vectors, contact_transforms, sealed, success, in_collision = (
#         suction_gripper.evaluate_grasps(
#             obj_mesh=obj_mesh,
#             points_on_surface=points_on_surface,
#             approach_vectors=approach_vectors,
#             grasp_transforms=grasp_transforms,
#             num_disturbances=num_disturbances,
#             qp_solver=qp_solver,
#             tqdm_disable=False,
#         )
#     )

#     return points, approach_vectors, contact_transforms, sealed, success, in_collision


# def create_grasp_dataset_json(
#     object_file,
#     object_scale,
#     contact_transforms,
#     sealed,
#     success,
#     in_collision,
#     tutorial_grasp_dir,
#     object_basename,
#     obj_center_mass,
# ):
#     """Create the grasp dataset JSON file following GraspGen convention."""

#     # Assert that obj_center_mass is a 3D vector
#     assert (
#         len(obj_center_mass) == 3
#     ), f"obj_center_mass should be length 3, got {len(obj_center_mass)}"

#     # Determine successful grasps (sealed, high success rate, and collision-free)
#     # We'll use a threshold for success rate to create a mix of positive and negative grasps
#     success_threshold = 0.5  # Grasps with >50% success rate are considered positive

#     # Create object_in_gripper mask
#     object_in_gripper = np.logical_and.reduce(
#         [
#             sealed,  # Must be sealed
#             # ~in_collision  # Must not be in collision; TODO - check if the collision mesh is specified correctly
#         ]
#     )

#     # Transform grasps back to original object frame (add back center of mass)
#     output_transforms = np.copy(contact_transforms)
#     output_transforms[:, :3, 3] += obj_center_mass

#     # Create the dataset dictionary
#     dataset_dict = {
#         "object": {
#             "file": object_file,  # Relative path to object in tutorial dataset
#             "scale": object_scale,
#         },
#         "grasps": {
#             "transforms": output_transforms.tolist(),
#             "object_in_gripper": object_in_gripper.tolist(),
#         },
#     }

#     # Save to JSON file
#     output_json_path = tutorial_grasp_dir / f"{object_basename}_grasps.json"
#     with open(output_json_path, "w") as f:
#         json.dump(dataset_dict, f, indent=2)

#     print(f"Grasp dataset saved to: {output_json_path}")
#     print(f"Total grasps: {len(object_in_gripper)}")
#     print(f"Positive grasps: {sum(object_in_gripper)}")
#     print(f"Negative grasps: {sum(~object_in_gripper)}")

#     return str(output_json_path)


# def create_splits_file(tutorial_object_dir, tutorial_grasp_dir, object_basename):
#     """Create train.txt and valid.txt files for the tutorial dataset."""

#     # Create train.txt (include the object)
#     train_txt_path = tutorial_object_dir / "train.txt"
#     with open(train_txt_path, "w") as f:
#         f.write(f"{object_basename}.obj\n")

#     # Create valid.txt (include the same object for validation)
#     valid_txt_path = tutorial_object_dir / "valid.txt"
#     with open(valid_txt_path, "w") as f:
#         f.write(f"{object_basename}.obj\n")

#     print(f"Split files created:")
#     print(f"  Train: {train_txt_path}")
#     print(f"  Valid: {valid_txt_path}")


# def main():
#     """Main function to generate the tutorial dataset."""
#     args = parse_args()

#     # Set random seed for reproducibility
#     np.random.seed(args.random_seed)

#     # Validate inputs
#     if not os.path.exists(args.object_path):
#         raise FileNotFoundError(f"Object file not found: {args.object_path}")

#     if not os.path.exists(args.gripper_config):
#         raise FileNotFoundError(f"Gripper config not found: {args.gripper_config}")

#     # Get object basename
#     object_basename = Path(args.object_path).stem

#     print(f"Generating dataset for object: {object_basename}")
#     print(f"Object path: {args.object_path}")
#     print(f"Output directory: {args.output_dir}")
#     print(f"Number of grasps: {args.num_grasps}")
#     print(f"Object scale: {args.object_scale}")

#     # Setup directories
#     tutorial_object_dir, tutorial_grasp_dir = setup_directories(
#         args.output_dir, object_basename
#     )
#     # Copy object to dataset
#     object_file, obj_mesh = copy_object_to_dataset(
#         args.object_path, tutorial_object_dir, object_basename, args.object_scale
#     )

#     # Make object_file a relative path with respect to tutorial_object_dir
#     object_file = f"{object_basename}.obj"

#     # Center the object at origin
#     obj_center_mass = obj_mesh.center_mass
#     obj_mesh.apply_translation(-obj_center_mass)
#     print(f"Object centered at origin. Original CoM: {obj_center_mass}")

#     # Initialize suction gripper
#     print(f"Loading gripper configuration: {args.gripper_config}")
#     suction_gripper = SuctionCupArray.from_file(fname=args.gripper_config)

#     # Generate grasps
#     points, approach_vectors, contact_transforms, sealed, success, in_collision = (
#         generate_grasps(
#             obj_mesh=obj_mesh,
#             suction_gripper=suction_gripper,
#             num_grasps=args.num_grasps,
#             num_disturbances=args.num_disturbances,
#             qp_solver=args.qp_solver,
#         )
#     )

#     # Create grasp dataset JSON
#     grasp_json_path = create_grasp_dataset_json(
#         object_file=object_file,
#         object_scale=args.object_scale,
#         contact_transforms=contact_transforms,
#         sealed=sealed,
#         success=success,
#         in_collision=in_collision,
#         tutorial_grasp_dir=tutorial_grasp_dir,
#         object_basename=object_basename,
#         obj_center_mass=obj_center_mass,
#     )

#     # Create split files
#     create_splits_file(tutorial_object_dir, tutorial_grasp_dir, object_basename)

#     # Optional visualization
#     if not args.no_visualization:
#         print("\nVisualizing results...")
#         from grasp_gen.utils.meshcat_utils import (
#             create_visualizer,
#             visualize_mesh,
#             visualize_pointcloud,
#             visualize_grasp,
#         )

#         vis = create_visualizer()

#         # Visualize object
#         visualize_mesh(vis, "object", obj_mesh, color=[169, 169, 169])

#         # Visualize grasp points
#         from grasp_gen.dataset.suction import colorize_for_meshcat

#         grasp_colors = colorize_for_meshcat(success)
#         visualize_pointcloud(vis, "grasp_points", points, grasp_colors, size=0.005)

#         # Visualize top 10 grasps
#         combined_criteria = sealed * success
#         top_indices = np.argsort(combined_criteria)[-10:][::-1]

#         for i, idx in enumerate(top_indices):
#             if combined_criteria[idx] > 0:
#                 color = [255, 0, 0] if i == 0 else [255, 165, 0]
#                 visualize_grasp(
#                     vis,
#                     f"top_grasps/grasp_{i:02d}",
#                     contact_transforms[idx],
#                     color=color,
#                     gripper_name="single_suction_cup_30mm",
#                 )

#         print("Visualization complete. Check meshcat browser window.")
#         input("Press Enter to continue...")

#     print(f"\nTutorial dataset generation complete!")
#     print(f"Object dataset: {tutorial_object_dir}")
#     print(f"Grasp dataset: {tutorial_grasp_dir}")
#     print(f"Ready for training generator and discriminator models!")


# if __name__ == "__main__":
#     main()
