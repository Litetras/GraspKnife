#!/usr/bin/env bash
set -euo pipefail

# Ablation: w/o Stage-I Region-Orientation Grasp Prior.
#
# Full LOD-Grasp uses Stage I CLIP strict phrase training to initialize the
# grasp generator/discriminator before Stage II Qwen natural-language alignment.
# This ablation keeps the Stage II Qwen-anchor training objective unchanged,
# but starts Stage II from random model weights instead of Stage I checkpoints.
#
# This tests whether the region-orientation prior learned in Stage I is an
# essential part of the two-stage framework.

export GRASPGEN_LANGUAGE_MODE=qwen_anchor
export LOD_LANGUAGE_MODE=qwen_anchor
export GRASPGEN_ABLATION_SUBSET="${GRASPGEN_ABLATION_SUBSET:-1}"

export TARGET="${TARGET:-both}"
export GRASPGEN_STAGE2_TRAIN_FULL_MODEL=1

export GEN_LOG_DIR="${GEN_LOG_DIR:-/results/tutorial/logs/ablation_stage2_no_stage1_prior_gen}"
export DIS_LOG_DIR="${DIS_LOG_DIR:-/results/tutorial/logs/ablation_stage2_no_stage1_prior_dis}"

if [[ -z "${GEN_CHECKPOINT:-}" && -f "$GEN_LOG_DIR/last.pth" ]]; then
    export GEN_CHECKPOINT="$GEN_LOG_DIR/last.pth"
    export GEN_RANDOM_INIT=0
else
    export GEN_RANDOM_INIT="${GEN_RANDOM_INIT:-1}"
fi

if [[ -z "${DIS_CHECKPOINT:-}" && -f "$DIS_LOG_DIR/last.pth" ]]; then
    export DIS_CHECKPOINT="$DIS_LOG_DIR/last.pth"
    export DIS_RANDOM_INIT=0
else
    export DIS_RANDOM_INIT="${DIS_RANDOM_INIT:-1}"
fi

# Keep the same 1000-epoch Stage II budget as the full Stage II run.
export GEN_NEPOCH="${GEN_NEPOCH:-1000}"
export DIS_NEPOCH="${DIS_NEPOCH:-1000}"

bash runs/train_stage2_qwen_align.sh
