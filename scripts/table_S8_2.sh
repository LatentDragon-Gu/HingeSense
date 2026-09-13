#!/usr/bin/env bash
# Table S8.2 (supplement): per-dataset single-channel read-out and hysteresis
# Table S8.2（补充材料）：逐数据集单通道读出与迟滞
# Usage:  [JOBS=<parallel processes>] [GPUS=<ids, e.g. 0,1>] [PYTHON=python] bash scripts/table_S8_2.sh
# 用法：  [JOBS=<并行进程数>] [GPUS=<GPU 编号，如 0,1>] [PYTHON=python] bash scripts/table_S8_2.sh
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
$PYTHON make_tables.py --only Table_S8_2
