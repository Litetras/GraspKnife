#!/usr/bin/env bash
set -euo pipefail

# Ablation: w/o Orientation Condition, Stage I CLIP strict prior.
# strict_text 从 "up handle" 退化为 "handle"，natural_text 也去掉方向词。

export GRASPGEN_NO_ORIENTATION_CONDITION=1
export GRASPGEN_ABLATION_SUBSET="${GRASPGEN_ABLATION_SUBSET:-1}"
export GRASPGEN_LANGUAGE_MODE=clip_strict
export LOD_LANGUAGE_MODE=clip_strict
export TARGET="${TARGET:-both}"
export GEN_LOG_DIR="${GEN_LOG_DIR:-/results/tutorial/logs/ablation_stage1_no_orientation}"
export GEN_CHECKPOINT="${GEN_CHECKPOINT:-$GEN_LOG_DIR/last.pth}"
export DIS_LOG_DIR="${DIS_LOG_DIR:-/results/tutorial/logs/ablation_stage1_no_orientation_dis}"
export DIS_CHECKPOINT="${DIS_CHECKPOINT:-$DIS_LOG_DIR/last.pth}"

bash runs/train_stage1_clip_prior.sh
