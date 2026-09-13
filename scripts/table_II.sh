#!/usr/bin/env bash
# Table II (main text): comparison methods vs. the proposed decoder
# Table II（正文）：对比方法与本文解码器
# Usage:  [JOBS=<parallel processes>] [GPUS=<ids, e.g. 0,1>] [PYTHON=python] bash scripts/table_II.sh
# 用法：  [JOBS=<并行进程数>] [GPUS=<GPU 编号，如 0,1>] [PYTHON=python] bash scripts/table_II.sh
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
$PYTHON train.py --method all --jobs "$JOBS" $GPU_ARG
$PYTHON make_tables.py --only Table_II
