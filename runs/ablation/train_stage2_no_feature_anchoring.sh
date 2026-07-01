#!/usr/bin/env bash
set -euo pipefail

# Ablation: w/o Feature Anchoring.
# Remove the CLIP affordance-phrase teacher, the feature anchoring loss,
# and the residual language adapter. Qwen natural-language features are
# projected to 512-D and directly condition the frozen Stage-I grasp prior.

export GRASPGEN_LANGUAGE_MODE=qwen_no_feature_anchor
export LOD_LANGUAGE_MODE=qwen_no_feature_anchor
export GRASPGEN_ABLATION_SUBSET="${GRASPGEN_ABLATION_SUBSET:-1}"
export GRASPGEN_DISABLE_FEATURE_ANCHORING=1
export GRASPGEN_DISABLE_CLIP_ANCHOR=1
export GRASPGEN_DISABLE_LANGUAGE_ADAPTER=1
export TARGET="${TARGET:-gen}"
export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-/results/tutorial/logs/stage1_clip_prior_gen/last.pth}"
export GEN_LOG_DIR="${GEN_LOG_DIR:-/results/tutorial/logs/ablation_stage2_no_feature_anchoring}"

bash runs/train_stage2_qwen_align.sh
