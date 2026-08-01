# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import os
import time
import numpy as np
import omegaconf
import torch
from pathlib import Path

from grasp_gen.dataset.dataset import collate
from grasp_gen.models.grasp_gen import GraspGen
from grasp_gen.models.m2t2 import M2T2
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal
from grasp_gen.utils.logging_config import get_logger

logger = get_logger(__name__)

def load_grasp_cfg(gripper_config: str) -> omegaconf.DictConfig:
    cfg = omegaconf.OmegaConf.load(gripper_config)
    ckpt_root_dir = Path(gripper_config).parent
    cfg.eval.checkpoint = str(ckpt_root_dir / cfg.eval.checkpoint)
    cfg.discriminator.checkpoint = str(ckpt_root_dir / cfg.discriminator.checkpoint)
    assert (
        cfg.data.gripper_name
        == cfg.diffusion.gripper_name
        == cfg.discriminator.gripper_name
    )
    return cfg

class GraspGenSampler:
    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = cfg
        if cfg.eval.model_name == "m2t2":
            model = M2T2.from_config(cfg.m2t2)
            ckpt = torch.load(cfg.eval.checkpoint)
            model.load_state_dict(ckpt["model"])
        elif cfg.eval.model_name == "diffusion-discriminator":
            model = GraspGen.from_config(cfg.diffusion, cfg.discriminator)
            if not os.path.exists(cfg.eval.checkpoint):
                raise FileNotFoundError(f"Checkpoint {cfg.eval.checkpoint} does not exist")
            if not os.path.exists(cfg.discriminator.checkpoint):
                raise FileNotFoundError(f"Checkpoint {cfg.discriminator.checkpoint} does not exist")

            model.load_state_dict(cfg.eval.checkpoint, cfg.discriminator.checkpoint)
            model.eval()
        else:
            raise NotImplementedError(f"Model name not implemented {cfg.eval.model_name}")

        self.model = model.cuda().eval()

    @staticmethod
    def run_inference(
        object_pc: np.ndarray | torch.Tensor,
        grasp_sampler: "GraspGenSampler",
        natural_text=None,
        strict_text=None,
        generator_natural_text=None,
        generator_strict_text=None,
        evaluator_natural_text=None,
        evaluator_strict_text=None,
        grasp_threshold: float = -1.0,
        num_grasps: int = 200,
        topk_num_grasps: int = -1,
        min_grasps: int = 40,
        max_tries: int = 6,
        remove_outliers: bool = True,
        return_raw: bool = False,
    ):

        if type(object_pc) == np.ndarray:
            object_pc = torch.from_numpy(object_pc).cuda().float()

        if grasp_threshold == -1.0 and topk_num_grasps == -1:
            topk_num_grasps = 100

        all_grasps = []
        all_conf = []
        all_raw_grasps = []
        all_raw_conf = []
        num_tries = 0

        while sum(len(g) for g in all_grasps) < min_grasps and num_tries < max_tries:
            num_tries += 1
            t0 = time.time()
            output = grasp_sampler.sample(
                object_pc,
                natural_text=natural_text,
                strict_text=strict_text,
                generator_natural_text=generator_natural_text,
                generator_strict_text=generator_strict_text,
                evaluator_natural_text=evaluator_natural_text,
                evaluator_strict_text=evaluator_strict_text,
                threshold=grasp_threshold,
                num_grasps=num_grasps,
                remove_outliers=remove_outliers,
                return_raw=return_raw,
            )
            if return_raw:
                raw_grasps = output[3]
                raw_conf = output[4]
                all_raw_grasps.append(raw_grasps)
                all_raw_conf.append(raw_conf)
            grasp_conf = output[1]
            grasps = output[0]

            if topk_num_grasps != -1 and len(grasps) > 0:
                grasp_conf, grasps = zip(
                    *sorted(zip(grasp_conf, grasps), key=lambda x: x[0], reverse=True)
                )
                grasps = torch.stack(grasps)
                grasp_conf = torch.stack(grasp_conf)
                grasps = grasps[:topk_num_grasps]
                grasp_conf = grasp_conf[:topk_num_grasps]

            all_grasps.append(grasps)
            all_conf.append(grasp_conf)
            t1 = time.time()

        if len(all_grasps) == 0:
            empty = torch.empty((0,), device=object_pc.device)
            if return_raw:
                return empty, empty, empty, empty
            return empty, empty

        grasps = torch.cat(all_grasps, dim=0)
        grasp_conf = torch.cat(all_conf, dim=0)

        # Rank the merged pool globally.  Previously Top-K was only applied
        # within each retry, so a low-scoring grasp from an early retry could
        # survive ahead of a better grasp generated later.
        if topk_num_grasps != -1 and len(grasps) > 0:
            order = torch.argsort(grasp_conf, descending=True)
            order = order[:topk_num_grasps]
            grasps = grasps[order]
            grasp_conf = grasp_conf[order]

        if os.environ.get("GRASPGEN_LOG_CONFIDENCE_STATS", "0") == "1":
            print(
                "[GraspGen confidence] "
                f"tries={num_tries}, merged_kept={len(grasps)}, "
                f"topk={topk_num_grasps}",
                flush=True,
            )
        grasps[:, 3, 3] = 1 

        if return_raw:
            raw_grasps = torch.cat(all_raw_grasps, dim=0)
            raw_conf = torch.cat(all_raw_conf, dim=0)
            raw_grasps[:, 3, 3] = 1
            return grasps, grasp_conf, raw_grasps, raw_conf

        return grasps, grasp_conf

    @torch.inference_mode()
    def sample(
        self,
        obj_pcd: np.ndarray,
        natural_text=None,
        strict_text=None,
        generator_natural_text=None,
        generator_strict_text=None,
        evaluator_natural_text=None,
        evaluator_strict_text=None,
        threshold: float = -1.0,
        num_grasps: int = 200,
        remove_outliers: bool = True,
        return_raw: bool = False,
    ):

        if remove_outliers:
            obj_pcd, _ = point_cloud_outlier_removal(obj_pcd)

        obj_pcd_center = obj_pcd.mean(axis=0)
        obj_pts_color = torch.zeros_like(obj_pcd)
        obj_mean_points = obj_pcd - obj_pcd_center[None]

        data = {}
        data["task"] = "pick"
        data["inputs"] = torch.cat([obj_mean_points, obj_pts_color[:, :3].squeeze(1)], dim=-1).float()
        data["points"] = obj_mean_points
        
        def unpack_text(value):
            return value[0] if isinstance(value, list) else value

        if natural_text is not None:
            data["natural_text"] = unpack_text(natural_text)
        if strict_text is not None:
            data["strict_text"] = unpack_text(strict_text)

        stage_texts = {
            "generator_natural_text": generator_natural_text,
            "generator_strict_text": generator_strict_text,
            "evaluator_natural_text": evaluator_natural_text,
            "evaluator_strict_text": evaluator_strict_text,
        }
        for key, value in stage_texts.items():
            if value is not None:
                data[key] = unpack_text(value)

        data_batch = collate([data])
        grasp_key = "grasps"
        with torch.inference_mode():
            if self.cfg.eval.model_name == "m2t2":
                model_outputs = self.model.infer(data_batch, self.cfg.eval)
            elif self.cfg.eval.model_name == "diffusion-discriminator":
                grasp_key = "grasps_pred"
                self.model.grasp_generator.num_grasps_per_object = num_grasps
                model_outputs, _, _ = self.model.infer(data_batch)
            else:
                raise NotImplementedError(f"Invalid model {self.cfg.eval.model_name}!")

        if len(model_outputs[grasp_key][0]) == 0:
            if return_raw:
                empty_pose = torch.empty((0, 4, 4), device=obj_pcd.device)
                empty_score = torch.empty((0,), device=obj_pcd.device)
                return empty_pose, empty_score, None, empty_pose, empty_score
            return [], [], []

        grasps = model_outputs[grasp_key][0]

        if self.cfg.eval.model_name == "diffusion-discriminator":
            grasp_conf_raw = model_outputs["grasp_confidence"][0][:, 0]
            raw_grasps = grasps.clone()
            raw_grasp_conf = grasp_conf_raw.clone()
            finite_mask = torch.isfinite(grasp_conf_raw)
            mask_best_grasps = finite_mask & (grasp_conf_raw >= threshold)

            if os.environ.get("GRASPGEN_LOG_CONFIDENCE_STATS", "0") == "1":
                finite_conf = grasp_conf_raw[finite_mask].detach().float()
                if finite_conf.numel() > 0:
                    quantiles = torch.quantile(
                        finite_conf,
                        torch.tensor(
                            [0.0, 0.5, 0.9, 1.0],
                            device=finite_conf.device,
                            dtype=finite_conf.dtype,
                        ),
                    ).cpu().tolist()
                    print(
                        "[GraspGen confidence] "
                        f"threshold={threshold:.6f}, raw={grasp_conf_raw.numel()}, "
                        f"finite={finite_conf.numel()}, kept={int(mask_best_grasps.sum())}, "
                        f"min={quantiles[0]:.6f}, median={quantiles[1]:.6f}, "
                        f"p90={quantiles[2]:.6f}, max={quantiles[3]:.6f}",
                        flush=True,
                    )
                else:
                    print(
                        "[GraspGen confidence] "
                        f"threshold={threshold:.6f}, raw={grasp_conf_raw.numel()}, "
                        "finite=0, kept=0",
                        flush=True,
                    )

            grasps = grasps[mask_best_grasps]
            grasp_conf = grasp_conf_raw[mask_best_grasps]

        elif self.cfg.eval.model_name == "m2t2":
            grasps = grasps[0]
            grasp_conf = model_outputs["grasp_confidence"][0][0]
            raw_grasps = grasps.clone()
            raw_grasp_conf = grasp_conf.clone()

        grasps[:, :3, 3] += obj_pcd_center
        if return_raw:
            raw_grasps[:, :3, 3] += obj_pcd_center
            return grasps, grasp_conf, None, raw_grasps, raw_grasp_conf
        return grasps, grasp_conf, None
