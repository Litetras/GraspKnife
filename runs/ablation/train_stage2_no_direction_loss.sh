#!/usr/bin/env bash
set -euo pipefail

# Ablation: w/o Direction Loss, Stage II Qwen alignment.

export GRASPGEN_DISABLE_DIRECTION_LOSS=1
export GRASPGEN_ABLATION_SUBSET="${GRASPGEN_ABLATION_SUBSET:-1}"
export GRASPGEN_LANGUAGE_MODE=qwen_anchor
export LOD_LANGUAGE_MODE=qwen_anchor
export TARGET="${TARGET:-gen}"
export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/results/tutorial/logs/ablation_stage1_no_direction_loss/last.pth}"
export GEN_LOG_DIR="${GEN_LOG_DIR:-/results/tutorial/logs/ablation_stage2_no_direction_loss}"

bash runs/train_stage2_qwen_align.sh
