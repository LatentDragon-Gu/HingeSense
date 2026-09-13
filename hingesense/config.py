# -*- coding: utf-8 -*-
"""全局常量、数据集编目与路径。所有常量为本系统的实验配置；修改任何常量都会改变输出结果。

Global constants, dataset catalogue and paths. Every constant is part of the experimental configuration;
changing any of them changes the results.
"""
import os

# ---------------------------------------------------------------- 路径 / paths
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PKG_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "Data")                      # 被测数据集 parquet（NoLoad / Assembly），程序直接读取，不生成中间文件 / evaluated datasets, read directly, no intermediate files
SUPP_DIR = os.path.join(ROOT_DIR, "Supplementary")             # 辅助数据集 parquet（NoLoad / Assembly），只作训练源 / auxiliary datasets, training sources only
RESULT_DIR = os.path.join(ROOT_DIR, "results")
RAW_DIR = os.path.join(RESULT_DIR, "raw")                      # 各程序的直接产物 / what the programs write
RANKING_FILE = os.path.join(RESULT_DIR, "Fig_S8_4", "electrode_ranking.txt")   # 电极重要性排名（electrode_importance.py 生成）/ electrode ranking written by electrode_importance.py

# ---------------------------------------------------------------- 数据集编目 / dataset catalogue
# 数据集名 = parquet 文件名 = 结果文件名。Data_01–Data_12 为文章编号的 12 个被测数据集，Data_aux_01–11 为仅作训练源的辅助数据集（Supplementary/）。
# 列表顺序即训练时的折表顺序，它参与随机数流，不可改动。
# Dataset name = parquet file name = result file name. Data_01–Data_12 are the 12 evaluated datasets numbered as in the paper,
# Data_aux_01–11 the auxiliary datasets used as training sources only (Supplementary/). The list order is the fold order used
# in training; it enters the random-number stream and must not be changed.
EVALUATED = [  # (数据集编号, 数据集名, 条件) / (dataset number, dataset name, condition)
    (1, "Data_01", "empty"),
    (2, "Data_02", "empty"),
    (3, "Data_03", "empty"),
    (4, "Data_04", "empty"),
    (5, "Data_05", "empty"),
    (6, "Data_06", "empty"),
    (12, "Data_12", "load"),
    (7, "Data_07", "load"),
    (8, "Data_08", "load"),
    (9, "Data_09", "load"),
    (11, "Data_11", "load"),
    (10, "Data_10", "load"),
]
AUXILIARY = [  # (数据集名, 条件) / (dataset name, condition)
    ("Data_aux_01", "empty"),
    ("Data_aux_02", "empty"),
    ("Data_aux_03", "empty"),
    ("Data_aux_04", "empty"),
    ("Data_aux_05", "empty"),
    ("Data_aux_06", "empty"),
    ("Data_aux_07", "empty"),
    ("Data_aux_08", "empty"),
    ("Data_aux_09", "load"),
    ("Data_aux_10", "load"),
    ("Data_aux_11", "load"),
]
FOLDS = [f for _, f, _ in EVALUATED]                         # 12 个被测数据集（训练折表顺序）/ the 12 evaluated datasets in training fold order
HOLDOUT = [f for f, _ in AUXILIARY]                           # 11 个辅助数据集 / the 11 auxiliary datasets
DATASET_NO = {f: n for n, f, _ in EVALUATED}                 # 数据集名 → 编号 / dataset name → number
FOLD_BY_NO = {n: f for n, f, _ in EVALUATED}
ORDER = sorted(FOLDS, key=lambda f: DATASET_NO[f])            # 按编号 1..12 排列（表格与命令行顺序）/ datasets ordered 1..12 (tables and command line)
COND = {f: c for _, f, c in EVALUATED}
COND.update({f: c for f, c in AUXILIARY})
LOAD_COND_FOLDS = {f for f, c in COND.items() if c == "load"}
COND_LABEL = {"empty": "no-load", "load": "assembly-constrained"}
COND_CLI = {"noload": "empty", "assembly": "load"}      # 命令行 --cond 的取值 → 内部条件代码 / --cond value → internal condition code
COND_NAME = {v: k for k, v in COND_CLI.items()}          # 内部条件代码 → 日志与输出中的名称 / internal code → name used in logs


def cond_of(fold_name):
    """数据集的物理条件；翻转变体（如 name@UD）与本体同条件。 / Mounting condition of a dataset; flipped copies (name@UD) share it."""
    return "load" if fold_name.split("@")[0] in LOAD_COND_FOLDS else "empty"


def parse_folds(spec):
    """命令行的数据集选择 → 数据集名列表（按编号排序）。接受编号（"1-12"、"1,3,7-9"）、数据集名（"Data_03"，逗号分隔）或 "all" / 空串。

    Command-line dataset selection → list of dataset names ordered by number. Accepts numbers ("1-12", "1,3,7-9"),
    names ("Data_03", comma-separated) or "all" / empty string.
    """
    if not spec or spec == "all":
        return list(ORDER)
    chosen = set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in DATASET_NO:
            chosen.add(tok)
        elif "-" in tok and all(p.strip().isdigit() for p in tok.split("-", 1)):
            a, b = (int(p) for p in tok.split("-", 1))
            for n in range(a, b + 1):
                if n not in FOLD_BY_NO:
                    raise ValueError("dataset No.%d does not exist (valid: 1-12)" % n)
                chosen.add(FOLD_BY_NO[n])
        elif tok.isdigit():
            if int(tok) not in FOLD_BY_NO:
                raise ValueError("dataset No.%s does not exist (valid: 1-12)" % tok)
            chosen.add(FOLD_BY_NO[int(tok)])
        else:
            raise ValueError("unknown dataset %r (use numbers 1-12 or names such as Data_01)" % tok)
    return [f for f in ORDER if f in chosen]


# ---------------------------------------------------------------- 原始数据字段 / raw data fields
N_ELECTRODE = 16
N_CHANNEL = 120                                                # C(16,2)
RAW_COLUMNS = ["xdat_%03d" % i for i in range(N_CHANNEL)]
ANGLE_COLUMN = "angle_deg"
SEGMENT_COLUMN = "xdat_segment"                                # 扫描周期编号 / scan-cycle index
SAMPLE_PERIOD_MS = 200.0                                       # 一次 120 通道扫描的周期 / period of one 120-channel scan
SUP_PCT = 0.3333                                               # 标定段占周期数比例 / calibration share of the cycles
VAL_PCT = 0.3333                                               # 验证段占周期数比例 / validation share of the cycles

# ---------------------------------------------------------------- 特征 / features
MASK_K = 40                # 全局高质量通道数 / channels kept by the global mask
DIFF_TAUS = (16, 64)       # 差分通路 EMA 半衰期（帧）/ EMA half-lives of the differential pathways (frames)
CLIP_Z = 8.0               # 稳健白化后的截断幅度 / clipping after robust whitening
CCN_HALFLIFE_MULT = 8.0    # 因果归一化参考的漂移中心半衰期 = 8 × 标定段长度 / drift-centre half-life = 8 × calibration length
SUP_ANGSTEP = 0.5          # 锚分组角度箱宽（度）/ angle-bin width for anchor grouping (deg)
SUP_K = 4                  # 每（角度箱 × 方向）组保留的锚帧数 / anchors kept per (angle bin × direction)
FLIP_TYPES = ("UD", "LR", "180")
SNR_BIN = 1.0              # 通道信噪比评分的角度箱宽（度）/ angle-bin width of the channel SNR score (deg)

# ---------------------------------------------------------------- 模型 / model
LAT = 64                   # 嵌入头输出维数 / embedding-head output size
HEAD_HID = 96              # 嵌入头隐层 / embedding-head hidden size
DRIFT_HID = 96             # DriftNet 隐层 / DriftNet hidden size
POOLN = 5                  # DriftNet 输入展开块数（cat[x, 0, x, x, x]）/ DriftNet input expansion blocks
FUSE_HID = 256             # 融合 MLP 隐层 / fusion MLP hidden size
FUSE_OUT = 96              # 最终嵌入维数 / final embedding size

# ---------------------------------------------------------------- 训练 / training
SEED = 2027
EPOCHS = 300
PATIENCE = 50
EPOCH_STEPS = 200
QBATCH = 256
LR = 1.5e-3
WD = 2e-4
CLIP_NORM = 2.0
AUXW = 0.25                # 辅助读出损失权重 / weight of the auxiliary read-out loss
HUBER_BETA = 2.0
RAMP_STD = 0.15            # 目标帧局部漂移噪声标准差 / std of the local drift noise on target frames
GAIN_STD = 0.12            # 共享通道增益扰动标准差 / std of the shared channel gain perturbation
OFFSET_STD = 0.25          # 共享通道偏置扰动标准差 / std of the shared channel offset perturbation
EVAL_BATCH = 1536

# ---------------------------------------------------------------- 消融 / ablations
# 消融配置名（命令行与结果目录均用此名）→ (特征 / 训练路径的内部开关, 通道掩膜数)。表格行标签见 ABLATION_LABELS。
# Ablation name (used on the command line and for the result folder) → (internal switch, channels kept by the mask). Row labels: ABLATION_LABELS.
ABLATIONS = {
    "no_drift_correction": ("nodrift", MASK_K),        # DriftNet 固定为恒等 / DriftNet fixed to identity
    "no_causal_differential": ("diff0", MASK_K),       # diff16 | diff64 通路置零 / differential pathways zeroed
    "no_channel_mask": ("", N_CHANNEL),                # 使用全部 120 通道 / all 120 channels
    "no_reference_pathway": ("ccn0", MASK_K),          # 因果归一化参考通路置零 / normalisation reference pathway zeroed
    "no_noise_perturbation": ("aug0", MASK_K),         # 去掉三种噪声 / 漂移扰动 / no noise-drift perturbation
    "no_symmetry_augmentation": ("flip0", MASK_K),     # 去掉电极环对称翻转副本 / no symmetry-flipped copies
    "no_augmentation": ("noaug", MASK_K),              # 同时去掉扰动与翻转 / neither perturbation nor flips
}
ABLATION_LABELS = {
    "decoder": "Complete system (16 electrodes)",
    "no_drift_correction": "w/o drift-correction network",
    "no_causal_differential": "w/o causal differential features",
    "no_channel_mask": "w/o channel mask (all 120 channels)",
    "no_reference_pathway": "w/o normalization reference pathway",
    "no_noise_perturbation": "w/o noise-drift perturbation",
    "no_symmetry_augmentation": "w/o geometric symmetry augmentation",
    "no_augmentation": "w/o all augmentation",
}

# ---------------------------------------------------------------- 对比方法 / comparison methods
BASELINE_METHODS = ("ridge", "rf", "kalman", "es", "gru", "tcn", "gru_aug", "tcn_aug", "rf_aug", "kalmannet",
                    "mlp", "calib_ridge", "rawknn", "transformer", "calib_kalman", "calib_es")
CPU_METHODS = ("ridge", "rf", "kalman", "es", "calib_ridge", "calib_kalman", "calib_es", "rf_aug")   # 不用 GPU 的方法 / methods that do not use a GPU
# 文章 Table II 的 13 种对比方法（--method all 的范围；运行顺序耗时长的在前，以缩短并行总时长）；三种增强变体只能按名单独运行
# The 13 comparison methods of Table II (scope of --method all; slowest first to shorten the parallel wall time); the three augmented variants run by name only
PAPER_METHODS = ("kalmannet", "gru", "tcn", "transformer", "mlp", "rawknn", "rf", "ridge", "kalman", "es", "calib_ridge", "calib_kalman", "calib_es")
AUGMENTED_METHODS = ("gru_aug", "tcn_aug", "rf_aug")

# ---------------------------------------------------------------- 评估 / evaluation
IMPORTANCE_SEEDS = 10                                          # 每电极的置换次数 / permutations per electrode
SUBSET_ELECTRODES = 4                                          # 电极精简配置保留的电极数（排名前 4）/ electrodes kept in the reduced configuration (top 4)
FAILURE_ELECTRODES = 3                                         # 单电极失效测试的电极数（排名前 3）/ electrodes tested for failure (top 3)
SHUFFLE_BLOCKS = (96, 128)                                     # 块乱序评测的块长 / block lengths of the block-shuffled evaluation
SHUFFLE_SEED_LABELS = (1, 2)                                   # 种子 = 100003 × 标签 + 31 / seed = 100003 × label + 31

# ---------------------------------------------------------------- 电极编号约定 / electrode numbering
# 代码全程使用 0 基编号 E0–E15；文档中的 1 基电极对（如 4–13）对应 E3–E12。
# The code uses zero-based electrodes E0–E15; one-based pair names in the paper (e.g. 4–13) denote E3–E12.
def channel_index(i, j):
    """0 基电极 i<j 的电极对 → 通道号（与采集枚举顺序一致）。 / Electrode pair i<j (zero-based) → channel index in acquisition order."""
    i, j = sorted((int(i), int(j)))
    return sum(N_ELECTRODE - 1 - a for a in range(i)) + (j - i - 1)


def electrodes_to_channels(electrodes):
    """电极集合 → 这些电极两两组成的通道号（升序）。 / Set of electrodes → sorted channel indices of all their pairs."""
    es = sorted(set(int(e) for e in electrodes))
    if any(e < 0 or e >= N_ELECTRODE for e in es):
        raise ValueError("electrode numbers must be within 0..%d" % (N_ELECTRODE - 1))
    return sorted(channel_index(a, b) for k, a in enumerate(es) for b in es[k + 1:])


def electrodes_tag(electrodes):
    """电极集合的目录名片段，如 E0-E1-E8-E15。 / Folder-name fragment of an electrode set, e.g. E0-E1-E8-E15."""
    return "-".join("E%d" % e for e in sorted(set(int(e) for e in electrodes)))
