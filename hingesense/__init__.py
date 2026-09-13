# -*- coding: utf-8 -*-
"""HingeSense：自感知 SCM 铰链的校准锚定角度解码。 / HingeSense: calibration-anchored angle decoding for a self-sensing SCM hinge.

模块 / modules:
    config     常量、数据集编目、路径 / constants, dataset catalogue, paths
    dataset    原始 parquet → 三段切分 → 因果归一化参考 → 通道掩膜 → 480 维帧特征 → 锚抽样 / parquet → segments → causal reference → channel mask → 480-D features → anchors
    model      DriftNet、嵌入头、门控融合、soft-kNN 读出 / DriftNet, embedding head, gated fusion, soft-kNN read-out
    trainer    留一数据集训练与评测（单折工作进程）/ leave-one-dataset-out training and evaluation (one dataset per process)
    baselines  16 种对比方法（单折工作进程）/ 16 comparison methods (one dataset per process)
    evaluate   单电极失效、置换重要性（单折工作进程）/ electrode failure and permutation importance (one dataset per process)
    ranking    电极重要性排名文件的写入与解析 / writer and parser of the electrode ranking file
    tables     原始结果 → 各表（终端打印 + csv）/ raw results → tables (terminal + csv)
    plots      Fig. S8.4
    cost       解码器批 1 时延与模型体量测量 / batch-1 latency and model size of the decoder
    runner     子进程并行调度与机器信息记录 / parallel job runner and machine information
"""
import os
import warnings

# cuBLAS 的确定性工作区必须在首次 CUDA 计算前设定；所有入口都先导入本包。
# The deterministic cuBLAS workspace must be set before the first CUDA call; every entry point imports this package first.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _format_warning(message, category, filename, lineno, line=None):
    """日志中的库警告只保留类别与内容，不写发出警告的文件路径。 / Library warnings in the logs keep category and message only, no file path."""
    return "%s: %s\n" % (category.__name__, message)


warnings.formatwarning = _format_warning

__version__ = "1.0.0"
