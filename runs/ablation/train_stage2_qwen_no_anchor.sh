#!/usr/bin/env bash
set -euo pipefail

# Ablation: Qwen-LoRA w/o CLIP Anchor.
# Qwen 仍然使用 natural_text，但不加载 CLIP teacher，也不产生 qwen_anchor_loss。

export GRASPGEN_LANGUAGE_MODE=qwen_no_anchor
export GRASPGEN_ABLATION_SUBSET="${GRASPGEN_ABLATION_SUBSET:-1}"
export LOD_LANGUAGE_MODE=qwen_no_anchor
export GRASPGEN_DISABLE_CLIP_ANCHOR=1
export TARGET="${TARGET:-gen}"
export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/results/tutorial/logs/stage1_clip_prior_gen/last.pth}"
export GEN_LOG_DIR="${GEN_LOG_DIR:-/results/tutorial/logs/ablation_stage2_qwen_no_anchor}"

bash runs/train_stage2_qwen_align.sh
