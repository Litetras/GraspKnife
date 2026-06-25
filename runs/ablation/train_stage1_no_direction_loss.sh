#!/usr/bin/env bash
set -euo pipefail

# Ablation: w/o Direction Loss, Stage I CLIP strict prior.
# 仍然记录 dir_acc_30，但 direction_loss 不加入总 loss。

export GRASPGEN_DISABLE_DIRECTION_LOSS=1
export GRASPGEN_ABLATION_SUBSET="${GRASPGEN_ABLATION_SUBSET:-1}"
export GRASPGEN_LANGUAGE_MODE=clip_strict
export LOD_LANGUAGE_MODE=clip_strict
export TARGET="${TARGET:-gen}"
export GEN_LOG_DIR="${GEN_LOG_DIR:-/results/tutorial/logs/ablation_stage1_no_direction_loss}"
export GEN_CHECKPOINT="${GEN_CHECKPOINT:-$GEN_LOG_DIR/last.pth}"

bash runs/train_stage1_clip_prior.sh
