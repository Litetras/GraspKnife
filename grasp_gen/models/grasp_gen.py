#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
该代码实现了一个端到端的抓取姿态生成与评估模型，核心定位是：

    功能整合：将扩散模型 - based 的抓取姿态生成器（GraspGenGenerator）和判别式抓取姿态评估器（GraspGenDiscriminator）整合为单一管道，解决 “生成 - 评估” 一体化问题；
    输入输出：以场景点云为核心输入，先生成候选抓取姿态，再通过判别器对生成的姿态进行有效性打分 / 评估，最终输出 “生成姿态 + 评估结果”；
    工程化适配：提供配置化实例化、预训练权重加载、统一推理接口等工程化特性，兼容 PyTorch 训练 / 推理框架，可直接集成到机器人抓取系统中；


    无训练逻辑：当前版本仅实现推理流程（forward/infer 无损失计算），核心用于部署阶段的抓取姿态生成与评估。

"""



import torch
import torch.nn as nn
from omegaconf import DictConfig

from grasp_gen.models.discriminator import GraspGenDiscriminator
from grasp_gen.models.generator import GraspGenGenerator
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


class GraspGen(nn.Module):
    """Combined model that uses both diffusion-based generation and discriminative evaluation.

    This class combines a GraspGenGenerator generator with a GraspGenDiscriminator to both
    generate and evaluate grasps in a single pipeline.

    Args:
        grasp_generator_cfg (DictConfig): Configuration for the grasp generator
        grasp_discriminator_cfg (DictConfig): Configuration for the grasp discriminator
    """

    def __init__(
        self, grasp_generator_cfg: DictConfig, grasp_discriminator_cfg: DictConfig
    ):
        super(GraspGen, self).__init__()
        self.grasp_generator = GraspGenGenerator.from_config(grasp_generator_cfg)
        self.grasp_discriminator = GraspGenDiscriminator.from_config(
            grasp_discriminator_cfg
        )

    def forward(self, data):
        """Forward pass combining generation and discrimination.

        Args:
            data: Input data dictionary containing point clouds

        Returns:
            tuple: (outputs, losses, stats) containing generated and scored grasps
        """
        outputs, _, stats = self.grasp_generator.infer(data, return_metrics=True)#生成器封装
        data.update(outputs)
        data["grasp_key"] = (
            "grasps_pred"  # Override to run discriminator inference on grasps predicted from previous step.#翻译：覆盖以运行判别器推理，使用前一步预测的抓取姿态
        )
        outputs, _, _ = self.grasp_discriminator.infer(data)#判别器封装
        return outputs, {}, stats

    def infer(self, data, return_metrics=False):
        """Inference method for generating and evaluating grasps.

        Args:
            data: Input data dictionary containing point clouds
            return_metrics (bool): Whether to compute evaluation metrics

        Returns:
            tuple: (outputs, losses, stats) containing generated and scored grasps with metrics
        """
        return self.forward(data)

    @classmethod
    def from_config(
        cls, grasp_generator_cfg: DictConfig, grasp_discriminator_cfg: DictConfig
    ):
        """Creates a GraspGen instance from configuration objects.

        Args:
            grasp_generator_cfg (DictConfig): Configuration for the grasp generator
            grasp_discriminator_cfg (DictConfig): Configuration for the grasp discriminator

        Returns:
            GraspGen: Instantiated model
        """
        return GraspGen(grasp_generator_cfg, grasp_discriminator_cfg)

    def load_state_dict(
        self, grasp_generator_ckpt_filepath: str, grasp_discriminator_ckpt_filepath: str
    ):
        """Loads pretrained weights for both generator and discriminator.

        Args:
            grasp_generator_ckpt_filepath (str): Path to generator checkpoint
            grasp_discriminator_ckpt_filepath (str): Path to discriminator checkpoint
        """
        logger.info(
            f"Loading generator checkpoint from {grasp_generator_ckpt_filepath}"
        )
        ckpt = torch.load(grasp_generator_ckpt_filepath, map_location="cpu")
        self.grasp_generator.load_state_dict(ckpt["model"], strict=False)###########################推理阶段，生成器允许部分权重不匹配（strict=False），以适应可能的架构差异或缺失的权重

        logger.info(
            f"Loading discriminator checkpoint from {grasp_discriminator_ckpt_filepath}"
        )
        ckpt = torch.load(grasp_discriminator_ckpt_filepath, map_location="cpu")

        self.grasp_discriminator.load_state_dict(ckpt["model"], strict=False)###########################推理阶段，判别器允许部分权重不匹配（strict=False），以适应可能的架构差异或缺失的权重
