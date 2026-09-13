#!/usr/bin/env bash
# Table IV (main text): ablation
# Table IV（正文）：消融
# Usage:  [JOBS=<parallel processes>] [GPUS=<ids, e.g. 0,1>] [PYTHON=python] bash scripts/table_IV.sh
# 用法：  [JOBS=<并行进程数>] [GPUS=<GPU 编号，如 0,1>] [PYTHON=python] bash scripts/table_IV.sh
# Existing results under results/raw are reused, so re-running only completes what is missing.
# results/raw 下已有的结果会被复用，重新运行只补齐缺少的部分。
set -e
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
JOBS=${JOBS:-1}
GPU_ARG=""
[ -n "${GPUS:-}" ] && GPU_ARG="--gpus $GPUS"

$PYTHON check_data.py
$PYTHON train.py --jobs "$JOBS" $GPU_ARG
$PYTHON train.py --ablation no_drift_correction --jobs "$JOBS" $GPU_ARG
$PYTHON train.py --ablation no_causal_differential --jobs "$JOBS" $GPU_ARG
$PYTHON train.py --ablation no_channel_mask --jobs "$JOBS" $GPU_ARG
$PYTHON train.py --ablation no_reference_pathway --jobs "$JOBS" $GPU_ARG
$PYTHON train.py --ablation no_noise_perturbation --jobs "$JOBS" $GPU_ARG
$PYTHON train.py --ablation no_symmetry_augmentation --jobs "$JOBS" $GPU_ARG
$PYTHON train.py --ablation no_augmentation --jobs "$JOBS" $GPU_ARG
$PYTHON make_tables.py --only Table_IV
