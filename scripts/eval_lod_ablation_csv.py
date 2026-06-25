#!/usr/bin/env python3
"""Minimal CSV evaluator for LOD-Grasp ablation runs.

This script intentionally stays light: it reuses the normal dataset/model path,
loads a checkpoint with strict=False, runs the requested split, and writes a CSV
with the fields needed by the ablation table. Metrics are filled only when the
model exposes matching stats; missing metrics are left blank instead of being
silently faked as zero.
"""

import csv
import os
from pathlib import Path

import hydra
import torch

from grasp_gen.models.grasp_gen import GraspGenDiscriminator, GraspGenGenerator
from grasp_gen.utils.train_utils import get_data_loader, to_gpu
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)


CSV_FIELDS = [
    "variant",
    "language_mode",
    "category",
    "task_name",
    "object_id",
    "scene",
    "strict_text",
    "natural_text",
    "gsr",
    "rsr",
    "osr30",
    "tsr",
    "dir_acc30",
    "num_candidates",
    "checkpoint",
]


def tensor_to_float(value):
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def first_list_value(value, default=""):
    if isinstance(value, list) and value:
        return value[0]
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        return int(value.reshape(-1)[0].detach().cpu().item())
    if value is None:
        return default
    return value


def load_checkpoint_state(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    incompatible = model.load_state_dict(state, strict=False)
    logger.info(
        "Loaded checkpoint strict=False: path=%s missing=%d unexpected=%d",
        checkpoint_path,
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )
    if incompatible.missing_keys:
        logger.info("Missing keys sample: %s", incompatible.missing_keys[:30])
    if incompatible.unexpected_keys:
        logger.info("Unexpected keys sample: %s", incompatible.unexpected_keys[:30])


def get_stat(stats, *keys):
    for key in keys:
        if key in stats:
            return tensor_to_float(stats[key])
    return ""


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):
    language_mode = os.environ.get(
        "GRASPGEN_LANGUAGE_MODE",
        os.environ.get("LOD_LANGUAGE_MODE", "qwen_anchor"),
    )
    variant = os.environ.get("GRASPGEN_ABLATION_VARIANT", language_mode)
    checkpoint = cfg.eval.checkpoint
    if not checkpoint:
        raise ValueError("eval.checkpoint must be set.")

    output_dir = Path(cfg.eval.output_dir or "./ablation_eval_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{variant}_{language_mode}_{cfg.eval.split}.csv"

    logger.info("========== LOD-Grasp Ablation Config ==========")
    logger.info("GRASPGEN_ABLATION_SUBSET = %s", os.environ.get("GRASPGEN_ABLATION_SUBSET", "0"))
    logger.info("GRASPGEN_LANGUAGE_MODE = %s", language_mode)
    logger.info("GRASPGEN_NO_ORIENTATION_CONDITION = %s", os.environ.get("GRASPGEN_NO_ORIENTATION_CONDITION", "0"))
    logger.info("GRASPGEN_DISABLE_DIRECTION_LOSS = %s", os.environ.get("GRASPGEN_DISABLE_DIRECTION_LOSS", "0"))
    logger.info("GRASPGEN_DISABLE_CLIP_ANCHOR = %s", os.environ.get("GRASPGEN_DISABLE_CLIP_ANCHOR", "0"))
    logger.info("GRASPGEN_DISABLE_SEMANTIC_NEGATIVES = %s", os.environ.get("GRASPGEN_DISABLE_SEMANTIC_NEGATIVES", "0"))
    logger.info("STAGE1_CHECKPOINT = %s", os.environ.get("STAGE1_CHECKPOINT", ""))
    logger.info("strict_text example = <logged from first batch>")
    logger.info("natural_text example = <logged from first batch>")
    logger.info("===============================================")

    _, loader = get_data_loader(
        cfg.train,
        cfg.data,
        cfg.eval.split,
        scenes=None,
        use_ddp=False,
        training=False,
    )

    if cfg.eval.model_name == "diffusion":
        model = GraspGenGenerator.from_config(cfg.diffusion).cuda().eval()
    elif cfg.eval.model_name == "discriminator":
        model = GraspGenDiscriminator.from_config(cfg.discriminator).cuda().eval()
    else:
        raise NotImplementedError(
            f"eval_lod_ablation_csv supports diffusion/discriminator, got {cfg.eval.model_name}"
        )

    load_checkpoint_state(model, checkpoint)

    max_batches = getattr(cfg.eval, "max_batches", None)
    rows = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            if data is None:
                continue

            strict_text = first_list_value(data.get("strict_text"))
            natural_text = first_list_value(data.get("natural_text"))
            task_name = first_list_value(data.get("pass_task_name"))
            object_id = first_list_value(data.get("object_ids"))
            scene = ""

            if batch_idx == 0:
                logger.info("strict_text example = %s", strict_text)
                logger.info("natural_text example = %s", natural_text)

            to_gpu(data)
            _, _, stats = model(data, cfg.train)
            stats_recon = {}
            if cfg.eval.model_name == "diffusion":
                _, outputs_recon, stats_recon = model(data, eval=True)
                num_candidates = tensor_to_float(
                    getattr(outputs_recon.get("grasps", None), "shape", ["", ""])[1]
                    if isinstance(outputs_recon, dict)
                    else ""
                )
            else:
                outputs_recon = {}
                num_candidates = ""

            merged_stats = {**stats, **stats_recon}
            rows.append(
                {
                    "variant": variant,
                    "language_mode": language_mode,
                    "category": "",
                    "task_name": task_name,
                    "object_id": object_id,
                    "scene": scene,
                    "strict_text": strict_text,
                    "natural_text": natural_text,
                    "gsr": get_stat(merged_stats, "gsr", "grasp_success_rate"),
                    "rsr": get_stat(merged_stats, "rsr", "region_success_rate"),
                    "osr30": get_stat(merged_stats, "osr30", "dir_acc_30"),
                    "tsr": get_stat(merged_stats, "tsr", "task_success_rate"),
                    "dir_acc30": get_stat(merged_stats, "dir_acc_30"),
                    "num_candidates": num_candidates,
                    "checkpoint": checkpoint,
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Ablation CSV saved: %s rows=%d", csv_path, len(rows))


if __name__ == "__main__":
    main()
