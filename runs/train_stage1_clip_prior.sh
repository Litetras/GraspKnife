#!/usr/bin/env bash
set -euo pipefail

# Stage I: train the CLIP strict-phrase region/orientation prior.
export LOD_LANGUAGE_MODE=clip_strict
export GRASPGEN_CACHE_TEXT_FEATURES="${GRASPGEN_CACHE_TEXT_FEATURES:-1}"

export NGPU="${NGPU:-1}"
export NWORKER="${NWORKER:-6}"
export NEPOCH="${NEPOCH:-6000}"
export BATCH="${BATCH:-8}"
export PRINT_FREQ="${PRINT_FREQ:-10}"
export PLOT_FREQ="${PLOT_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-1000}"
export DATASET_NAME="${DATASET_NAME:-objaverse}"
export DATASET_VERSION="${DATASET_VERSION:-v2}"
export TIMESTEPS="${TIMESTEPS:-10}"
export NUM_GRASPS_PER_OBJ="${NUM_GRASPS_PER_OBJ:-500}"
export BACKBONE="${BACKBONE:-pointnet}"
export NUM_POINTS="${NUM_POINTS:-3500}"
export P_PC="${P_PC:-0.50}"
export RED="${RED:-14}"
export GRIPPER_NAME="${GRIPPER_NAME:-franka_panda}"
export CODE_DIR="${CODE_DIR:-/code}"
export RESULTS_DIR="${RESULTS_DIR:-/results/tutorial}"
export OBJECT_DATASET_DIR="${OBJECT_DATASET_DIR:-$RESULTS_DIR/tutorial_object_dataset}"
export GRASP_DIR="${GRASP_DIR:-$RESULTS_DIR/tutorial_grasp_dataset}"
export GRASP_DATASET_DIR="${GRASP_DATASET_DIR:-$GRASP_DIR}"
export SPLIT_DATASET_DIR="${SPLIT_DATASET_DIR:-$OBJECT_DATASET_DIR}"
export ROTATION_REPR="${ROTATION_REPR:-r3_so3}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export NOISE_SCALE="${NOISE_SCALE:-1.0}"
export LOG_DIR="${LOG_DIR:-$RESULTS_DIR/logs/stage1_clip_prior}"
export CHECKPOINT="${CHECKPOINT:-$LOG_DIR/last.pth}"
export CACHE_DIR="${CACHE_DIR:-$RESULTS_DIR/cache}"

echo "Running Stage I CLIP strict prior for $GRIPPER_NAME"
mkdir -p "$LOG_DIR" "$CACHE_DIR" "$RESULTS_DIR"
cp "$CODE_DIR/tutorials/natural_texts.json" "$RESULTS_DIR/natural_texts.json" 2>/dev/null || true

cd "$CODE_DIR"
pip install -e . --no-deps
cd "$CODE_DIR/scripts"

python train_graspgen.py \
    data.num_points="$NUM_POINTS" \
    data.load_contact=False \
    data.dataset_cls="ObjectPickDataset" \
    data.rotation_augmentation=True \
    data.cache_dir="$CACHE_DIR" \
    data.root_dir="$SPLIT_DATASET_DIR" \
    data.object_root_dir="$OBJECT_DATASET_DIR" \
    data.grasp_root_dir="$GRASP_DATASET_DIR" \
    data.dataset_name="$DATASET_NAME" \
    data.dataset_version="$DATASET_VERSION" \
    data.prob_point_cloud="$P_PC" \
    data.redundancy="$RED" \
    data.gripper_name="$GRIPPER_NAME" \
    train.log_dir="$LOG_DIR" \
    train.batch_size="$BATCH" \
    train.num_gpus="$NGPU" \
    train.num_epochs="$NEPOCH" \
    train.num_workers="$NWORKER" \
    train.print_freq="$PRINT_FREQ" \
    train.plot_freq="$PLOT_FREQ" \
    train.save_freq="$SAVE_FREQ" \
    train.checkpoint="$CHECKPOINT" \
    train.model_name='diffusion' \
    train.debug=True \
    optimizer.type="ADAMW" \
    optimizer.lr=0.00001 \
    optimizer.grad_clip=-1 \
    diffusion.gripper_name="$GRIPPER_NAME" \
    diffusion.num_diffusion_iters="$TIMESTEPS" \
    diffusion.num_diffusion_iters_eval="$TIMESTEPS" \
    diffusion.obs_backbone="$BACKBONE" \
    diffusion.grasp_repr="$ROTATION_REPR" \
    diffusion.attention='cat_attn' \
    diffusion.compositional_schedular=True \
    diffusion.loss_pointmatching=False \
    diffusion.loss_l1_pos=True \
    diffusion.loss_l1_rot=True \
    diffusion.ptv3.grid_size=0.01 \
    diffusion.pose_repr='mlp' \
    diffusion.kappa="$NOISE_SCALE" \
    data.num_grasps_per_object="$NUM_GRASPS_PER_OBJ" \
    data.load_discriminator_dataset=False \
    data.visualize_batch=False \
    +diffusion.use_language_conditioning=True \
    +diffusion.clip_backbone="ViT-B/32" \
    +diffusion.lang_proj_dim=512
