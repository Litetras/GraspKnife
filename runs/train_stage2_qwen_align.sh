#!/usr/bin/env bash
set -euo pipefail

# Stage II: Qwen-LoRA natural-language alignment with frozen CLIP strict prior.
# 默认训练 generator + discriminator；如需单独跑：
#   TARGET=gen bash runs/train_stage2_qwen_align.sh
#   TARGET=dis bash runs/train_stage2_qwen_align.sh
export LOD_LANGUAGE_MODE=qwen_anchor
export GRASPGEN_LANGUAGE_MODE="${GRASPGEN_LANGUAGE_MODE:-qwen_anchor}"
export GRASPGEN_CACHE_TEXT_FEATURES="${GRASPGEN_CACHE_TEXT_FEATURES:-1}"
export GRASPGEN_CACHE_SAVE_FREQ="${GRASPGEN_CACHE_SAVE_FREQ:-100}"
export GRASPGEN_QWEN_CACHE_TOKENIZER="${GRASPGEN_QWEN_CACHE_TOKENIZER:-1}"
export GRASPGEN_QWEN_DEDUP_BATCH="${GRASPGEN_QWEN_DEDUP_BATCH:-1}"
export GRASPGEN_QWEN_USE_BACKBONE_ONLY="${GRASPGEN_QWEN_USE_BACKBONE_ONLY:-1}"
export GRASPGEN_QWEN_GRADIENT_CHECKPOINTING="${GRASPGEN_QWEN_GRADIENT_CHECKPOINTING:-0}"
export GRASPGEN_QWEN_MAX_LENGTH="${GRASPGEN_QWEN_MAX_LENGTH:-64}"
export IGNORE_OBJECT_CATEGORIES="${IGNORE_OBJECT_CATEGORIES:-screwdriver}"
export GRASPGEN_ABLATION_SUBSET="${GRASPGEN_ABLATION_SUBSET:-0}"
export GRASPGEN_ABLATION_SUBSET_CATEGORIES="${GRASPGEN_ABLATION_SUBSET_CATEGORIES:-knife,hammer,brush,spoon}"

export TARGET="${TARGET:-both}"
export NGPU="${NGPU:-1}"
export NWORKER="${NWORKER:-6}"
export PRINT_FREQ="${PRINT_FREQ:-10}"
export DATASET_NAME="${DATASET_NAME:-objaverse}"
export DATASET_VERSION="${DATASET_VERSION:-v2}"
export BACKBONE="${BACKBONE:-pointnet}"
export POSE_REPR="${POSE_REPR:-mlp}"
export ROTATION_REPR="${ROTATION_REPR:-r3_so3}"
export TOPK_RATIO="${TOPK_RATIO:-0.75}"
export NOISE_SCALE="${NOISE_SCALE:-1.0}"
export P_PC="${P_PC:-0.50}"
export RED="${RED:-14}"
export GRIPPER_NAME="${GRIPPER_NAME:-franka_panda}"
export CODE_DIR="${CODE_DIR:-/code}"
export RESULTS_DIR="${RESULTS_DIR:-/results/tutorial}"
export OBJECT_DATASET_DIR="${OBJECT_DATASET_DIR:-$RESULTS_DIR/tutorial_object_dataset}"
export GRASP_DIR="${GRASP_DIR:-$RESULTS_DIR/tutorial_grasp_dataset}"
export GRASP_DATASET_DIR="${GRASP_DATASET_DIR:-$GRASP_DIR}"
export SPLIT_DATASET_DIR="${SPLIT_DATASET_DIR:-$OBJECT_DATASET_DIR}"
export CACHE_DIR="${CACHE_DIR:-$RESULTS_DIR/cache}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

is_truthy() {
    case "${1:-0}" in
        1|true|True|TRUE|yes|Yes|YES|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

# Stage II is language alignment on top of the Stage I prior; empirically it
# converges much faster than Stage I, so keep the default short.
export STAGE2_RANDOM_INIT="${STAGE2_RANDOM_INIT:-0}"
export GEN_NEPOCH="${GEN_NEPOCH:-1000}"
export GEN_BATCH="${GEN_BATCH:-8}"
export GEN_PLOT_FREQ="${GEN_PLOT_FREQ:-10}"
export GEN_SAVE_FREQ="${GEN_SAVE_FREQ:-1000}"
export GEN_TIMESTEPS="${GEN_TIMESTEPS:-10}"
export GEN_NUM_GRASPS_PER_OBJ="${GEN_NUM_GRASPS_PER_OBJ:-500}"
export GEN_NUM_POINTS="${GEN_NUM_POINTS:-3500}"
export GEN_LR="${GEN_LR:-0.00001}"
export GEN_LOG_DIR="${GEN_LOG_DIR:-$RESULTS_DIR/logs/stage2_qwen_align_gen}"
export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-$RESULTS_DIR/logs/stage1_clip_prior_gen/last.pth}"
export STAGE1_GEN_CHECKPOINT="${STAGE1_GEN_CHECKPOINT:-$STAGE1_CHECKPOINT}"
export STAGE2_GEN_CHECKPOINT="${STAGE2_GEN_CHECKPOINT:-$GEN_LOG_DIR/last.pth}"
export GEN_RANDOM_INIT="${GEN_RANDOM_INIT:-$STAGE2_RANDOM_INIT}"
if is_truthy "$GEN_RANDOM_INIT"; then
    export GEN_CHECKPOINT="${GEN_CHECKPOINT:-}"
    export GEN_RESET_EPOCH_ON_LOAD="${GEN_RESET_EPOCH_ON_LOAD:-False}"
elif [[ -z "${GEN_CHECKPOINT:-}" ]]; then
    if [[ -f "$STAGE2_GEN_CHECKPOINT" ]]; then
        export GEN_CHECKPOINT="$STAGE2_GEN_CHECKPOINT"
        export GEN_RESET_EPOCH_ON_LOAD="${GEN_RESET_EPOCH_ON_LOAD:-False}"
    else
        export GEN_CHECKPOINT="$STAGE1_GEN_CHECKPOINT"
        export GEN_RESET_EPOCH_ON_LOAD="${GEN_RESET_EPOCH_ON_LOAD:-True}"
    fi
else
    if [[ "$GEN_CHECKPOINT" == "$STAGE1_GEN_CHECKPOINT" ]]; then
        export GEN_RESET_EPOCH_ON_LOAD="${GEN_RESET_EPOCH_ON_LOAD:-True}"
    else
        export GEN_RESET_EPOCH_ON_LOAD="${GEN_RESET_EPOCH_ON_LOAD:-False}"
    fi
fi

export DIS_NEPOCH="${DIS_NEPOCH:-1000}"
export DIS_BATCH="${DIS_BATCH:-16}"
export DIS_PLOT_FREQ="${DIS_PLOT_FREQ:-10}"
export DIS_SAVE_FREQ="${DIS_SAVE_FREQ:-1000}"
export DIS_NUM_GRASPS_PER_OBJ="${DIS_NUM_GRASPS_PER_OBJ:-300}"
export DIS_NUM_POINTS="${DIS_NUM_POINTS:-2048}"
export DIS_LR="${DIS_LR:-0.00001}"
export DIS_RATIO="${DIS_RATIO:-[0.50,0.45,0.00,0.05,0.00,0.00,0.00]}"
export DIS_LOG_DIR="${DIS_LOG_DIR:-$RESULTS_DIR/logs/stage2_qwen_align_dis}"
export STAGE1_DIS_CHECKPOINT="${STAGE1_DIS_CHECKPOINT:-$RESULTS_DIR/logs/stage1_clip_prior_dis/last.pth}"
export STAGE2_DIS_CHECKPOINT="${STAGE2_DIS_CHECKPOINT:-$DIS_LOG_DIR/last.pth}"
export DIS_RANDOM_INIT="${DIS_RANDOM_INIT:-$STAGE2_RANDOM_INIT}"
if is_truthy "$DIS_RANDOM_INIT"; then
    export DIS_CHECKPOINT="${DIS_CHECKPOINT:-}"
    export DIS_RESET_EPOCH_ON_LOAD="${DIS_RESET_EPOCH_ON_LOAD:-False}"
elif [[ -z "${DIS_CHECKPOINT:-}" ]]; then
    if [[ -f "$STAGE2_DIS_CHECKPOINT" ]]; then
        export DIS_CHECKPOINT="$STAGE2_DIS_CHECKPOINT"
        export DIS_RESET_EPOCH_ON_LOAD="${DIS_RESET_EPOCH_ON_LOAD:-False}"
    else
        export DIS_CHECKPOINT="$STAGE1_DIS_CHECKPOINT"
        export DIS_RESET_EPOCH_ON_LOAD="${DIS_RESET_EPOCH_ON_LOAD:-True}"
    fi
else
    if [[ "$DIS_CHECKPOINT" == "$STAGE1_DIS_CHECKPOINT" ]]; then
        export DIS_RESET_EPOCH_ON_LOAD="${DIS_RESET_EPOCH_ON_LOAD:-True}"
    else
        export DIS_RESET_EPOCH_ON_LOAD="${DIS_RESET_EPOCH_ON_LOAD:-False}"
    fi
fi

validate_dataset_paths() {
    local missing=0

    for required_file in "$SPLIT_DATASET_DIR/train.txt" "$SPLIT_DATASET_DIR/valid.txt"; do
        if [[ ! -f "$required_file" ]]; then
            echo "❌ Missing dataset split file: $required_file" >&2
            missing=1
        fi
    done

    for required_dir in "$OBJECT_DATASET_DIR" "$GRASP_DATASET_DIR"; do
        if [[ ! -d "$required_dir" ]]; then
            echo "❌ Missing dataset directory: $required_dir" >&2
            missing=1
        fi
    done

    if [[ "$missing" -ne 0 ]]; then
        cat >&2 <<EOF

Dataset path check failed.
Current paths:
  RESULTS_DIR=$RESULTS_DIR
  SPLIT_DATASET_DIR=$SPLIT_DATASET_DIR
  OBJECT_DATASET_DIR=$OBJECT_DATASET_DIR
  GRASP_DATASET_DIR=$GRASP_DATASET_DIR

If you are running inside Docker, make sure the new dataset is mounted to /results/tutorial.
Or override paths explicitly, for example:
  RESULTS_DIR=/results/tutorial \\
  OBJECT_DATASET_DIR=/path/to/tutorial_object_dataset \\
  GRASP_DIR=/path/to/tutorial_grasp_dataset \\
  SPLIT_DATASET_DIR=/path/to/tutorial_object_dataset \\
  bash runs/train_stage2_qwen_align.sh
EOF
        exit 2
    fi
}

prepare_env() {
    validate_dataset_paths
    mkdir -p "$RESULTS_DIR" "$CACHE_DIR"
    cp "$CODE_DIR/tutorials/natural_texts.json" "$RESULTS_DIR/natural_texts.json" 2>/dev/null || true
    cd "$CODE_DIR"
    pip install -e . --no-deps
    cd "$CODE_DIR/scripts"
}

run_generator() {
    echo "Running Stage II generator: Qwen anchor alignment"
    if [[ -n "$GEN_CHECKPOINT" ]]; then
        echo "Generator checkpoint: $GEN_CHECKPOINT"
    else
        echo "Generator checkpoint: <none, random init>"
    fi
    echo "Generator reset_epoch_on_load: $GEN_RESET_EPOCH_ON_LOAD"
    mkdir -p "$GEN_LOG_DIR"
    python train_graspgen.py \
        data.num_points="$GEN_NUM_POINTS" \
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
        train.log_dir="$GEN_LOG_DIR" \
        train.batch_size="$GEN_BATCH" \
        train.num_gpus="$NGPU" \
        train.num_epochs="$GEN_NEPOCH" \
        train.num_workers="$NWORKER" \
        train.print_freq="$PRINT_FREQ" \
        train.plot_freq="$GEN_PLOT_FREQ" \
        train.save_freq="$GEN_SAVE_FREQ" \
        train.checkpoint="$GEN_CHECKPOINT" \
        ++train.reset_epoch_on_load="$GEN_RESET_EPOCH_ON_LOAD" \
        train.model_name='diffusion' \
        train.debug=True \
        optimizer.type="ADAMW" \
        optimizer.lr="$GEN_LR" \
        optimizer.grad_clip=-1 \
        diffusion.gripper_name="$GRIPPER_NAME" \
        diffusion.num_diffusion_iters="$GEN_TIMESTEPS" \
        diffusion.num_diffusion_iters_eval="$GEN_TIMESTEPS" \
        diffusion.obs_backbone="$BACKBONE" \
        diffusion.grasp_repr="$ROTATION_REPR" \
        diffusion.attention='cat_attn' \
        diffusion.compositional_schedular=True \
        diffusion.loss_pointmatching=False \
        diffusion.loss_l1_pos=True \
        diffusion.loss_l1_rot=True \
        diffusion.ptv3.grid_size=0.01 \
        diffusion.pose_repr="$POSE_REPR" \
        diffusion.kappa="$NOISE_SCALE" \
        data.num_grasps_per_object="$GEN_NUM_GRASPS_PER_OBJ" \
        data.load_discriminator_dataset=False \
        data.visualize_batch=False \
        diffusion.use_language_conditioning=True \
        diffusion.clip_backbone="ViT-B/32" \
        diffusion.lang_proj_dim=512
}

run_discriminator() {
    echo "Running Stage II discriminator: Qwen anchor alignment"
    if [[ -n "$DIS_CHECKPOINT" ]]; then
        echo "Discriminator checkpoint: $DIS_CHECKPOINT"
    else
        echo "Discriminator checkpoint: <none, random init>"
    fi
    echo "Discriminator reset_epoch_on_load: $DIS_RESET_EPOCH_ON_LOAD"
    mkdir -p "$DIS_LOG_DIR"
    python train_graspgen.py \
        data.num_points="$DIS_NUM_POINTS" \
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
        train.log_dir="$DIS_LOG_DIR" \
        train.batch_size="$DIS_BATCH" \
        train.num_gpus="$NGPU" \
        train.num_epochs="$DIS_NEPOCH" \
        train.num_workers="$NWORKER" \
        train.print_freq="$PRINT_FREQ" \
        train.plot_freq="$DIS_PLOT_FREQ" \
        train.save_freq="$DIS_SAVE_FREQ" \
        train.checkpoint="$DIS_CHECKPOINT" \
        ++train.reset_epoch_on_load="$DIS_RESET_EPOCH_ON_LOAD" \
        train.model_name='discriminator' \
        train.debug=True \
        optimizer.type="ADAMW" \
        optimizer.grad_clip=-1 \
        optimizer.lr="$DIS_LR" \
        discriminator.gripper_name="$GRIPPER_NAME" \
        discriminator.topk_ratio="$TOPK_RATIO" \
        discriminator.obs_backbone="$BACKBONE" \
        discriminator.grasp_repr="$ROTATION_REPR" \
        discriminator.pose_repr="$POSE_REPR" \
        discriminator.kappa="$NOISE_SCALE" \
        discriminator.ptv3.grid_size=0.01 \
        data.num_grasps_per_object="$DIS_NUM_GRASPS_PER_OBJ" \
        data.load_discriminator_dataset=True \
        data.discriminator_ratio="$DIS_RATIO" \
        ++discriminator.use_language_conditioning=True \
        ++discriminator.clip_backbone="ViT-B/32" \
        ++discriminator.lang_proj_dim=512
}

prepare_env
case "$TARGET" in
    gen) run_generator ;;
    dis) run_discriminator ;;
    both) run_generator; run_discriminator ;;
    *) echo "Unknown TARGET=$TARGET, expected gen/dis/both"; exit 2 ;;
esac
