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



import os

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

    @staticmethod
    def _language_view(data, stage):
        """Return a shallow data view with optional stage-specific text inputs.

        Existing callers keep using ``natural_text`` and ``strict_text``.  Evidence-chain
        experiments may additionally provide ``generator_natural_text`` / ``generator_strict_text``
        and ``evaluator_natural_text`` / ``evaluator_strict_text`` so proposal generation and
        post-hoc scoring can be varied independently without changing any other input.
        """
        stage_data = dict(data)
        for text_key in ("natural_text", "strict_text"):
            override_key = f"{stage}_{text_key}"
            if override_key in data:
                stage_data[text_key] = data[override_key]
        return stage_data

    def forward(self, data):
        """Forward pass combining generation and discrimination.

        Args:
            data: Input data dictionary containing point clouds

        Returns:
            tuple: (outputs, losses, stats) containing generated and scored grasps
        """
        generator_data = self._language_view(data, "generator")
        outputs, _, stats = self.grasp_generator.infer(
            generator_data, return_metrics=True
        )  # 生成器封装

        # Preserve the original caller-visible update while allowing the evaluator
        # to receive a different text view of the same generated candidates.
        data.update(outputs)
        evaluator_data = self._language_view(data, "evaluator")
        evaluator_data["grasp_key"] = (
            "grasps_pred"  # Override to run discriminator inference on grasps predicted from previous step.#翻译：覆盖以运行判别器推理，使用前一步预测的抓取姿态
        )
        outputs, _, _ = self.grasp_discriminator.infer(evaluator_data)  # 判别器封装
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
        self._load_component_checkpoint(
            self.grasp_generator,
            grasp_generator_ckpt_filepath,
            component_name="generator",
        )
        self._load_component_checkpoint(
            self.grasp_discriminator,
            grasp_discriminator_ckpt_filepath,
            component_name="evaluator",
        )

    @staticmethod
    def _checkpoint_audit_enabled():
        return os.environ.get("GRASPGEN_STRICT_CHECKPOINT_AUDIT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

    @staticmethod
    def _allowed_unexpected_prefixes(module, component_name):
        """Return checkpoint-only language modules allowed by an intervention.

        The formal Direct CLIP-Natural ablation deliberately instantiates the
        Stage-I CLIP language path from the full Stage-II checkpoint.  Qwen and
        residual-adapter tensors therefore remain in the checkpoint but are not
        part of the CLIP-Natural module graph.  No other mismatch is acceptable.
        """
        language_mode = getattr(module, "language_mode", "none")
        if language_mode not in {"clip_natural", "clip_strict"}:
            return ()
        prefixes = ["qwen_text_encoder."]
        if component_name == "generator":
            prefixes.append("language_adapter.")
        return tuple(prefixes)

    @staticmethod
    def _is_qwen_quantization_metadata(key):
        """Identify bitsandbytes NF4 sidecars saved beside Qwen parameters.

        Recent bitsandbytes versions attach this state directly to quantized
        parameter objects instead of exposing it as ordinary ``state_dict``
        entries.  A checkpoint written by an older version can therefore report
        these sidecars as unexpected even though the corresponding quantized
        weight is present and loaded.  Keep this allowlist deliberately narrow:
        only Qwen keys with known NF4 metadata suffixes are accepted.
        """
        if not key.startswith("qwen_text_encoder."):
            return False
        return key.endswith(
            (
                ".absmax",
                ".quant_map",
                ".nested_absmax",
                ".nested_quant_map",
                ".quant_state.bitsandbytes__nf4",
            )
        )

    @classmethod
    def _load_component_checkpoint(cls, module, checkpoint_path, component_name):
        logger.info("Loading %s checkpoint from %s", component_name, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" not in checkpoint:
            raise KeyError(f"Checkpoint has no 'model' state: {checkpoint_path}")

        incompatible = module.load_state_dict(checkpoint["model"], strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        language_mode = getattr(module, "language_mode", "none")

        if not cls._checkpoint_audit_enabled():
            if missing or unexpected:
                logger.warning(
                    "Legacy non-strict checkpoint load: component=%s mode=%s "
                    "missing=%d unexpected=%d",
                    component_name,
                    language_mode,
                    len(missing),
                    len(unexpected),
                )
            return

        allowed_prefixes = cls._allowed_unexpected_prefixes(module, component_name)
        invalid_unexpected = [
            key
            for key in unexpected
            if not any(key.startswith(prefix) for prefix in allowed_prefixes)
            and not cls._is_qwen_quantization_metadata(key)
        ]

        required_prefixes = []
        if language_mode in {"clip_natural", "clip_strict"}:
            required_prefixes.extend(("clip_text_encoder.", "clip_text_projection."))
        elif language_mode in {"qwen_anchor", "qwen_no_anchor", "qwen_no_feature_anchor"}:
            required_prefixes.append("qwen_text_encoder.")
            if component_name == "generator" and getattr(
                module, "use_language_adapter", False
            ):
                required_prefixes.append("language_adapter.")

        state_keys = tuple(checkpoint["model"].keys())
        absent_required_prefixes = [
            prefix
            for prefix in required_prefixes
            if not any(key.startswith(prefix) for key in state_keys)
        ]

        if missing or invalid_unexpected or absent_required_prefixes:
            raise RuntimeError(
                "Audited checkpoint load failed: "
                f"component={component_name}, mode={language_mode}, "
                f"missing={missing[:30]}, "
                f"invalid_unexpected={invalid_unexpected[:30]}, "
                f"absent_required_prefixes={absent_required_prefixes}, "
                f"allowed_unexpected_prefixes={allowed_prefixes}, "
                f"checkpoint={checkpoint_path}"
            )

        logger.info(
            "Audited checkpoint load passed: component=%s mode=%s "
            "missing=0 unexpected=%d allowed_unexpected_prefixes=%s "
            "(remaining unexpected keys are audited NF4 metadata)",
            component_name,
            language_mode,
            len(unexpected),
            allowed_prefixes,
        )
