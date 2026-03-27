# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# 使用方法：
# 1. 启动meshcat-server
# 2. 运行脚本并指定点云数据目录和夹爪配置文件
# 示例：python scripts/demo_scene_pc.py --filter_collisions --sample_data_dir /models/sample_data/real_scene_pc --gripper_config /models/checkpoints/graspgen_franka_panda.yml

# 导入必要的库
import argparse  # 命令行参数解析
import glob  # 文件路径匹配
import json  # JSON数据处理
import os  # 操作系统功能
import omegaconf  # 配置文件处理

import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
import trimesh.transformations as tra  # 三维变换矩阵处理
from IPython import embed  # 交互式调试

# 从GraspGen库导入必要模块
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg  # 抓取生成采样器和配置加载
from grasp_gen.utils.meshcat_utils import (  # 可视化相关工具
    create_visualizer,  # 创建可视化器
    get_color_from_score,  # 根据分数获取颜色（用于可视化区分）
    get_normals_from_mesh,  # 从网格获取法向量
    make_frame,  # 创建坐标系可视化
    visualize_grasp,  # 可视化抓取姿态
    visualize_mesh,  # 可视化网格模型
    visualize_pointcloud,  # 可视化点云
)
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal  # 点云离群点去除


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="在单个物体点云上可视化GraspGen推理后的抓取姿态"
    )
    parser.add_argument(
        "--sample_data_dir",
        type=str,
        default="/home/zyp/GraspGen/models/sample_data/real_object_pc",
        help="包含点云数据JSON文件的目录"
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default="/home/zyp/Desktop/zyp_dataset6/tutorial/models/tutorial_model_config.yaml",
        help="夹爪配置YAML文件的路径"
    )
    parser.add_argument(
        "--grasp_threshold",
        type=float,
        default=0.8,
        help="有效抓取的阈值。若为-1.0，则返回排名前100的抓取姿态"
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=200,
        help="生成的抓取姿态数量"
    )
    parser.add_argument(
        "--return_topk",
        action="store_true",
        help="是否仅返回排名前k的抓取姿态"
    )
    parser.add_argument(
        "--topk_num_grasps",
        type=int,
        default=1,#default=-1,####################################################ZYP
        help="当return_topk为True时，返回的前k个抓取姿态数量"
    )

    # ==== 新增语言文本参数 ====###########
    parser.add_argument(
        "--text",
        type=str,
        default="up", 
        help="语言控制指令，例如 'up', 'down', 'top', 'low'"
    )
    # ========================############

    return parser.parse_args()


def process_point_cloud(pc, grasps, grasp_conf):
    """
    处理点云和抓取姿态，将它们居中（以点云中心为原点）

    参数:
        pc: 原始点云数据 (N, 3)
        grasps: 原始抓取姿态矩阵 (M, 4, 4)
        grasp_conf: 抓取姿态的置信度分数 (M,)

    返回:
        pc_centered: 居中后的点云
        grasps_centered: 居中后的抓取姿态
        scores: 基于置信度的颜色值（用于可视化）
    """
    # 根据置信度分数生成颜色（分数越高，颜色越绿；越低越红）
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    print(f"抓取置信度范围: 最小值 {grasp_conf.min():.3f}, 最大值 {grasp_conf.max():.3f}")

    # 确保抓取姿态矩阵是齐次坐标形式（最后一个元素为1）
    grasps[:, 3, 3] = 1

    # 计算点云中心，生成平移矩阵（将点云中心移到原点）
    T_subtract_pc_mean = tra.translation_matrix(-pc.mean(axis=0))
    # 对点云应用平移，实现居中
    pc_centered = tra.transform_points(pc, T_subtract_pc_mean)
    # 对所有抓取姿态应用相同的平移，保持相对位置不变
    grasps_centered = np.array(
        [T_subtract_pc_mean @ np.array(g) for g in grasps.tolist()]
    )

    return pc_centered, grasps_centered, scores


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 检查夹爪配置文件是否有效
    if args.gripper_config == "":
        raise ValueError("必须提供夹爪配置文件路径")
    if not os.path.exists(args.gripper_config):
        raise ValueError(f"夹爪配置文件不存在: {args.gripper_config}")

    # 处理topk参数逻辑：若启用return_topk且未指定数量，则默认返回前100
    if args.return_topk and args.topk_num_grasps == -1:
        args.topk_num_grasps = 100

    # 获取数据目录中所有JSON文件（每个文件对应一个物体的点云数据）
    json_files = glob.glob(os.path.join(args.sample_data_dir, "*.json"))
    # 创建可视化器（基于meshcat）
    vis = create_visualizer()

    # 加载夹爪配置文件（包含夹爪参数、模型路径等）
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    # 获取夹爪名称（用于可视化对应的夹爪模型）
    gripper_name = grasp_cfg.data.gripper_name

    # 初始化抓取生成采样器（只需初始化一次，可复用）
    grasp_sampler = GraspGenSampler(grasp_cfg)

    # 遍历所有点云数据文件
    for json_file in json_files:
        print(f"正在处理文件: {json_file}")
        # 清除当前可视化内容，准备显示新物体
        vis.delete()

        # 从JSON文件加载数据
        data = json.load(open(json_file, "rb"))
        pc = np.array(data["pc"])  # 点云坐标 (N, 3)
        pc_color = np.array(data["pc_color"])  # 点云颜色 (N, 3)
        grasps = np.array(data["grasp_poses"])  # 抓取姿态矩阵 (M, 4, 4)
        grasp_conf = np.array(data["grasp_conf"])  # 抓取置信度 (M,)

        # 处理点云和抓取姿态：居中处理
        pc_centered, grasps_centered, scores = process_point_cloud(
            pc, grasps, grasp_conf
        )

        # 可视化居中后的原始点云（大小为0.0025）
        visualize_pointcloud(vis, "pc", pc_centered, pc_color, size=0.0025)

        # 对点云进行离群点去除（减少噪声影响）
        pc_filtered, pc_removed = point_cloud_outlier_removal(
            torch.from_numpy(pc_centered)  # 转换为PyTorch张量处理
        )
        pc_filtered = pc_filtered.numpy()  # 过滤后的点云（保留有效点）
        pc_removed = pc_removed.numpy()  # 被过滤的离群点
        # 可视化离群点（红色，稍大尺寸0.003，便于区分）
        visualize_pointcloud(vis, "pc_removed", pc_removed, [255, 0, 0], size=0.003)

        # 使用过滤后的点云运行GraspGen推理，生成抓取姿态
        grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
            pc_filtered,  # 输入点云
            grasp_sampler,  # 抓取采样器实例
            grasp_threshold=args.grasp_threshold,  # 抓取置信度阈值
            num_grasps=args.num_grasps,  # 生成的抓取数量
            topk_num_grasps=args.topk_num_grasps, # 若启用，返回前k个最优抓取
            text=args.text  # <==== 新增这一行，将文字指令传给底层的模型##############
        )

        # 若生成了有效抓取姿态，则进行可视化
        if len(grasps_inferred) > 0:
            # 将结果从GPU（若使用）转移到CPU并转换为numpy数组
            grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
            grasps_inferred = grasps_inferred.cpu().numpy()
            # 确保抓取姿态是齐次坐标形式
            grasps_inferred[:, 3, 3] = 1
            # ================= 新增：将抓取姿态旋转 90 度 =================###########
            print("\n[注意]：生成的点云抓取姿态已绕局部 Z 轴旋转了 90 度！\n")
            R_90 = tra.rotation_matrix(np.pi / 2, [0, 0, 1])
            grasps_inferred = np.array([g @ R_90 for g in grasps_inferred])
            # ==============================================================#########
            # 根据置信度生成颜色（用于可视化区分不同质量的抓取）
            scores_inferred = get_color_from_score(
                grasp_conf_inferred, use_255_scale=True
            )
            print(
                f"推理生成 {len(grasps_inferred)} 个抓取姿态，置信度范围: {grasp_conf_inferred.min():.3f} - {grasp_conf_inferred.max():.3f}"
            )

            # 逐个可视化抓取姿态
            for j, grasp in enumerate(grasps_inferred):
                visualize_grasp(
                    vis,  # 可视化器实例
                    f"grasps_objectpc_filtered/{j:03d}/grasp",  # 可视化路径（用于区分不同抓取）
                    grasp,  # 抓取姿态矩阵
                    color=scores_inferred[j],  # 颜色（基于置信度）
                    gripper_name=gripper_name,  # 夹爪名称（加载对应模型）
                    linewidth=0.6,  # 可视化线宽
                )

        else:
            print("推理未生成任何有效抓取姿态！跳过当前物体...")

        # 等待用户输入，按Enter键继续处理下一个物体
        input("按Enter键继续处理下一个物体...")
