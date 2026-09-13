#!/usr/bin/env bash
# Table III (main text): representative single-channel read-out vs. the decoder
# Table III（正文）：代表单通道读出与解码器对比
# Usage:  [JOBS=<parallel processes>] [GPUS=<ids, e.g. 0,1>] [PYTHON=python] bash scripts/table_III.sh
# 用法：  [JOBS=<并行进程数>] [GPUS=<GPU 编号，如 0,1>] [PYTHON=python] bash scripts/table_III.sh
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
$PYTHON make_tables.py --only Table_III
