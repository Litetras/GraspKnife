# 适配GraspGen项目的MeshCat可视化npz抓取结果文件

import argparse
import os
import numpy as np
import trimesh.transformations as tra
from IPython import embed

# 导入GraspGen的MeshCat可视化工具（核心依赖）
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_pointcloud,
    visualize_grasp,
)
from grasp_gen.grasp_server import load_grasp_cfg  # 加载夹爪配置


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用MeshCat可视化collision_free_grasps_results.npz文件"
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default="/home/zyp/IsaacLab/collision_free_grasps_results.npz",
        help="npz结果文件的路径"
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default="/home/zyp/GraspGen/models/checkpoints/graspgen_robotiq_2f_140.yml",
        help="夹爪配置YAML文件路径（用于加载夹爪3D模型）"
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.0025,
        help="点云可视化的点大小（单位：米）"
    )
    parser.add_argument(
        "--grasp_linewidth",
        type=float,
        default=1.0,
        help="抓取姿态可视化的线宽"
    )
    parser.add_argument(
        "--max_pc_points",
        type=int,
        default=10000,
        help="点云下采样的最大数量（避免卡顿）"
    )
    return parser.parse_args()


def downsample_point_cloud(pc, max_points):
    """点云下采样（随机采样，保持均匀性）"""
    if len(pc) <= max_points:
        return pc
    indices = np.random.choice(len(pc), max_points, replace=False)
    return pc[indices]


def process_point_cloud(pc, grasps, grasp_conf):
    """
    点云和抓取姿态居中处理（与原项目逻辑一致）
    :param pc: 原始点云 (N, 3)
    :param grasps: 原始抓取姿态 (M, 4, 4)
    :param grasp_conf: 抓取置信度 (M,)
    :return: 居中后的点云、抓取姿态、抓取分数对应的颜色、点云中心
    """
    # ========== 核心修复1：空抓取分数防护 ==========
    if len(grasp_conf) == 0:
        print("⚠️  无碰撞抓取分数为空，跳过分数计算")
        scores = np.zeros((0, 3))  # 空颜色数组
        pc_mean = pc.mean(axis=0) if len(pc) > 0 else np.zeros(3)
        pc_centered = pc - pc_mean if len(pc) > 0 else pc
        grasps_centered = grasps  # 空抓取姿态直接返回
        return pc_centered, grasps_centered, scores, pc_mean

    # 根据分数生成颜色（原项目的配色逻辑：分数越高越绿）
    scores = get_color_from_score(grasp_conf, use_255_scale=True)
    print(f"抓取分数范围: 最小值 {grasp_conf.min():.3f}, 最大值 {grasp_conf.max():.3f}")

    # 确保抓取姿态是齐次矩阵
    grasps[:, 3, 3] = 1

    # 点云居中（以点云中心为原点）
    pc_mean = pc.mean(axis=0)
    T_subtract_mean = tra.translation_matrix(-pc_mean)
    pc_centered = tra.transform_points(pc, T_subtract_mean)

    # 抓取姿态居中：矩阵乘法（关键修复！不是transform_points）
    grasps_centered = np.array([T_subtract_mean @ g for g in grasps.tolist()])

    return pc_centered, grasps_centered, scores, pc_mean


if __name__ == "__main__":
    # 1. 解析参数
    args = parse_args()

    # 2. 校验文件有效性
    if not os.path.exists(args.npz_path):
        raise ValueError(f"npz文件不存在: {args.npz_path}")
    if not os.path.exists(args.gripper_config):
        raise ValueError(f"夹爪配置文件不存在: {args.gripper_config}")

    # 3. 初始化MeshCat可视化器（需先启动meshcat-server）
    print("📌 请确保已启动meshcat-server，浏览器访问 http://localhost:7000 查看可视化（注意端口：7000）")
    vis = create_visualizer()
    vis.delete()  # 清空历史可视化内容

    # 4. 加载并解析npz文件
    results = np.load(args.npz_path, allow_pickle=True)
    print(f"✅ 加载npz文件成功，包含数据集：{results.files}")

    # 解析核心数据
    object_pc = results["object_pc"]  # 目标物体点云 (P, 3)
    scene_pc = results["scene_pc"]  # 场景点云 (Q, 3)
    all_grasps = results["all_grasps"]  # 所有抓取姿态 (N, 4, 4)
    all_scores = results["all_scores"]  # 所有抓取分数 (N,)
    collision_free_mask = results["collision_free_mask"]  # 无碰撞掩码 (N,)
    collision_free_grasps = results["collision_free_grasps"]  # 无碰撞抓取 (M, 4, 4)
    collision_free_scores = results["collision_free_scores"]  # 无碰撞分数 (M,)

    # 5. 加载夹爪配置（用于可视化夹爪模型）
    grasp_cfg = load_grasp_cfg(args.gripper_config)
    gripper_name = grasp_cfg.data.gripper_name
    print(f"🔧 使用夹爪模型: {gripper_name}")

    # 6. 点云下采样（避免数量过大导致卡顿）
    object_pc = downsample_point_cloud(object_pc, args.max_pc_points)
    scene_pc = downsample_point_cloud(scene_pc, args.max_pc_points)

    # 7. 数据统计
    print(f"\n📊 数据统计（下采样后）：")
    print(f"  - 物体点云数量: {len(object_pc)}")
    print(f"  - 场景点云数量: {len(scene_pc)}")
    print(f"  - 生成抓取总数: {len(all_grasps)}")
    print(f"  - 无碰撞抓取数: {len(collision_free_grasps)}")

    # 8. 数据预处理（居中）
    pc_mean = np.zeros(3)  # 默认中心
    object_pc_centered = object_pc
    collision_free_grasps_centered = collision_free_grasps
    scores_inferred = np.zeros((0, 3))

    # ========== 核心修复2：仅当无碰撞抓取数>0时调用处理函数 ==========
    if len(object_pc) > 0 and len(collision_free_grasps) > 0:
        # 仅对有数据的无碰撞抓取做居中
        object_pc_centered, collision_free_grasps_centered, scores_inferred, pc_mean = process_point_cloud(
            object_pc, collision_free_grasps, collision_free_scores
        )
    elif len(object_pc) == 0:
        print("⚠️  物体点云为空，跳过居中处理")
    else:
        print("⚠️  无碰撞抓取数为0，跳过抓取姿态居中处理")

    # 9. 可视化点云
    # 9.1 场景点云（灰色，若有）
    if len(scene_pc) > 0:
        scene_pc_centered = tra.transform_points(scene_pc, tra.translation_matrix(-pc_mean))
        visualize_pointcloud(
            vis, "scene_pc", scene_pc_centered,
            color=[128, 128, 128],  # 灰色
            size=args.point_size
        )
        print("🎨 已可视化场景点云（灰色）")

    # 9.2 物体点云（绿色）
    if len(object_pc) > 0:
        visualize_pointcloud(
            vis, "object_pc", object_pc_centered,
            color=[0, 255, 0],  # 绿色
            size=args.point_size
        )
        print("🎨 已可视化物体点云（绿色）")

    # 10. 可视化抓取姿态
    # 10.1 无碰撞抓取（彩色，分数越高越绿）
    if len(collision_free_grasps_centered) > 0:
        print(f"🎯 可视化 {len(collision_free_grasps_centered)} 个无碰撞抓取姿态...")
        for j, (grasp, color) in enumerate(zip(collision_free_grasps_centered, scores_inferred)):
            try:
                visualize_grasp(
                    vis,
                    f"collision_free_grasps/{j:03d}",  # MeshCat中的路径（区分不同抓取）
                    grasp,  # 抓取姿态矩阵
                    color=color,  # 分数对应的颜色
                    gripper_name=gripper_name,  # 加载夹爪3D模型
                    linewidth=args.grasp_linewidth
                )
            except Exception as e:
                print(f"⚠️  可视化第{j}个无碰撞抓取失败: {e}")
                continue
    else:
        print("🚫 无碰撞抓取数为0，跳过无碰撞抓取可视化")

    # 10.2 碰撞抓取（红色，可选，仅显示前20个）
    colliding_grasps = all_grasps[~collision_free_mask] if len(all_grasps) > 0 else np.array([])
    if len(colliding_grasps) > 0:
        # 关键修复：抓取姿态居中用矩阵乘法，而非transform_points
        T_subtract_mean = tra.translation_matrix(-pc_mean)
        colliding_grasps_centered = np.array([T_subtract_mean @ g for g in colliding_grasps.tolist()])

        show_num = min(20, len(colliding_grasps_centered))
        print(f"🚫 可视化 {show_num} 个碰撞抓取姿态（红色，共{len(colliding_grasps)}个）...")
        for j in range(show_num):
            try:
                visualize_grasp(
                    vis,
                    f"colliding_grasps/{j:03d}",
                    colliding_grasps_centered[j],
                    color=[255, 0, 0],  # 红色标记碰撞抓取
                    gripper_name=gripper_name,
                    linewidth=args.grasp_linewidth * 0.8  # 线宽稍小，区分无碰撞
                )
            except Exception as e:
                print(f"⚠️  可视化第{j}个碰撞抓取失败: {e}")
                continue
    else:
        print("🚫 无碰撞抓取掩码为空/无碰撞抓取数为0，跳过碰撞抓取可视化")

    # 11. 保持可视化（等待用户输入后退出）
    print("\n✅ 可视化完成！")
    print("   - 浏览器访问 http://localhost:7000 查看3D场景（日志显示端口是7000）")
    print("   - 鼠标操作：左键旋转 | 滚轮缩放 | 右键平移")
    input("按Enter键退出可视化...")
    vis.delete()  # 退出前清空场景
