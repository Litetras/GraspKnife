#!/usr/bin/env bash
set -euo pipefail

# Ablation: w/o Orientation Condition, Stage II Qwen alignment.

export GRASPGEN_NO_ORIENTATION_CONDITION=1
export GRASPGEN_ABLATION_SUBSET="${GRASPGEN_ABLATION_SUBSET:-1}"
export GRASPGEN_LANGUAGE_MODE=qwen_anchor
export LOD_LANGUAGE_MODE=qwen_anchor
export TARGET="${TARGET:-gen}"
export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/results/tutorial/logs/ablation_stage1_no_orientation/last.pth}"
export GEN_LOG_DIR="${GEN_LOG_DIR:-/results/tutorial/logs/ablation_stage2_no_orientation}"
export GEN_CHECKPOINT="${GEN_CHECKPOINT:-$STAGE1_CHECKPOINT}"

bash runs/train_stage2_qwen_align.sh
