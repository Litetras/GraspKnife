#!/usr/bin/env bash
set -euo pipefail

# Optional ablation: w/o Semantic-Negative Selector.
# 只影响 discriminator 数据加载里的 opposite semantic hard negatives。

export GRASPGEN_DISABLE_SEMANTIC_NEGATIVES=1
export GRASPGEN_ABLATION_SUBSET="${GRASPGEN_ABLATION_SUBSET:-1}"
export GRASPGEN_LANGUAGE_MODE=clip_strict
export LOD_LANGUAGE_MODE=clip_strict
export TARGET="${TARGET:-dis}"
export DIS_LOG_DIR="${DIS_LOG_DIR:-/results/tutorial/logs/ablation_dis_no_semantic_negatives}"
export DIS_CHECKPOINT="${DIS_CHECKPOINT:-$DIS_LOG_DIR/last.pth}"

bash runs/train_stage1_clip_prior.sh
