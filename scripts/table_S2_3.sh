#!/usr/bin/env bash
# Table S2.3 (supplement): batch-size-one decoder latency on this machine
# Table S2.3（补充材料）：本机上的批 1 解码时延
# Usage:  [JOBS=<parallel processes>] [GPUS=<ids, e.g. 0,1>] [PYTHON=python] bash scripts/table_S2_3.sh
# 用法：  [JOBS=<并行进程数>] [GPUS=<GPU 编号，如 0,1>] [PYTHON=python] bash scripts/table_S2_3.sh
# Existing results under results/raw are reused, so re-running only completes what is missing.
# results/raw 下已有的结果会被复用，重新运行只补齐缺少的部分。
set -e
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
JOBS=${JOBS:-1}
GPU_ARG=""
[ -n "${GPUS:-}" ] && GPU_ARG="--gpus $GPUS"

$PYTHON check_data.py
$PYTHON train.py --folds 1 --jobs "$JOBS" $GPU_ARG
$PYTHON inference_cost.py
$PYTHON make_tables.py --only Table_S2_3
