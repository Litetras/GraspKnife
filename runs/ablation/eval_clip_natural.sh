#!/usr/bin/env bash
set -euo pipefail

# Ablation: CLIP-only Natural Instruction.
# 使用 Stage I CLIP prior checkpoint，但把语言输入从 strict_text 改为 natural_text。

export GRASPGEN_LANGUAGE_MODE=clip_natural
export LOD_LANGUAGE_MODE=clip_natural
export GRASPGEN_ABLATION_VARIANT=clip_natural
export GRASPGEN_ABLATION_SUBSET="${GRASPGEN_ABLATION_SUBSET:-1}"
export GRASPGEN_ABLATION_SUBSET_CATEGORIES="${GRASPGEN_ABLATION_SUBSET_CATEGORIES:-knife,hammer,brush,spoon}"
export IGNORE_OBJECT_CATEGORIES="${IGNORE_OBJECT_CATEGORIES:-screwdriver}"

export CODE_DIR="${CODE_DIR:-/code}"
export RESULTS_DIR="${RESULTS_DIR:-/results/tutorial}"
export OBJECT_DATASET_DIR="${OBJECT_DATASET_DIR:-$RESULTS_DIR/tutorial_object_dataset}"
export GRASP_DIR="${GRASP_DIR:-$RESULTS_DIR/tutorial_grasp_dataset}"
export GRASP_DATASET_DIR="${GRASP_DATASET_DIR:-$GRASP_DIR}"
export SPLIT_DATASET_DIR="${SPLIT_DATASET_DIR:-$OBJECT_DATASET_DIR}"
export CACHE_DIR="${CACHE_DIR:-$RESULTS_DIR/cache}"
export CHECKPOINT="${CHECKPOINT:-$RESULTS_DIR/logs/stage1_clip_prior_gen/last.pth}"
export OUTPUT_DIR="${OUTPUT_DIR:-$RESULTS_DIR/eval/ablation_clip_natural}"

mkdir -p "$OUTPUT_DIR"
cd "$CODE_DIR/scripts"

python eval_lod_ablation_csv.py \
    data.num_points="${NUM_POINTS:-3500}" \
    data.load_contact=False \
    data.dataset_cls="ObjectPickDataset" \
    data.rotation_augmentation=False \
    data.cache_dir="$CACHE_DIR" \
    data.root_dir="$SPLIT_DATASET_DIR" \
    data.object_root_dir="$OBJECT_DATASET_DIR" \
    data.grasp_root_dir="$GRASP_DATASET_DIR" \
    data.dataset_name="${DATASET_NAME:-objaverse}" \
    data.dataset_version="${DATASET_VERSION:-v2}" \
    data.prob_point_cloud="${P_PC:-0.50}" \
    data.redundancy="${RED:-14}" \
    data.gripper_name="${GRIPPER_NAME:-franka_panda}" \
    data.num_grasps_per_object="${NUM_GRASPS_PER_OBJ:-500}" \
    data.load_discriminator_dataset=False \
    train.batch_size="${EVAL_BATCH:-4}" \
    train.num_workers="${NWORKER:-4}" \
    train.model_name='diffusion' \
    diffusion.gripper_name="${GRIPPER_NAME:-franka_panda}" \
    diffusion.num_diffusion_iters="${TIMESTEPS:-10}" \
    diffusion.num_diffusion_iters_eval="${TIMESTEPS:-10}" \
    diffusion.obs_backbone="${BACKBONE:-pointnet}" \
    diffusion.grasp_repr="${ROTATION_REPR:-r3_so3}" \
    diffusion.attention='cat_attn' \
    diffusion.compositional_schedular=True \
    diffusion.loss_pointmatching=False \
    diffusion.loss_l1_pos=True \
    diffusion.loss_l1_rot=True \
    diffusion.ptv3.grid_size=0.01 \
    diffusion.pose_repr="${POSE_REPR:-mlp}" \
    diffusion.kappa="${NOISE_SCALE:-1.0}" \
    diffusion.use_language_conditioning=True \
    diffusion.clip_backbone="ViT-B/32" \
    diffusion.lang_proj_dim=512 \
    eval.model_name='diffusion' \
    eval.split="${SPLIT:-valid}" \
    eval.checkpoint="$CHECKPOINT" \
    eval.output_dir="$OUTPUT_DIR" \
    eval.exp_name='ablation_clip_natural' \
    eval.batch_size="${EVAL_BATCH:-4}" \
    eval.max_num_grasps="${MAX_NUM_GRASPS:-200}" \
    eval.num_seed_grasps="${NUM_SEED_GRASPS:-150}"
