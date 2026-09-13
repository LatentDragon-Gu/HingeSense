# -*- coding: utf-8 -*-
"""数据集：原始 parquet → 三段切分 → 因果归一化参考 → 通道掩膜 → 480 维帧特征 → 锚抽样。

数据流：
    load_raw           读取一个数据集的 parquet（120 通道电阻 + 视觉角度 + 周期编号），按周期数 1/3:1/3:1/3 切分（进程内缓存）
    causal_adaptive_norm  每通道一阶因果自适应归一化（drift 参考通路，S2.1-B）
    compute_channel_mask  全部数据集标定段的信噪比 → 全局 top-K 通道（S2.1-C）
    load_fold          [rawz | ccnz | diff16 | diff64] 四条 120 维通路拼成 480 维帧特征（S2.1-A/D）
    support_indices    标定段按 0.5° 角度箱 × 运动方向分组，每组均匀保留 4 帧作为锚（S2.1-E）
不生成任何中间数据文件。

Datasets: raw parquet → three segments → causal normalisation reference → channel mask → 480-D frame features → anchors.
    load_raw           read one dataset (120 resistances + vision angle + cycle index), split 1/3:1/3:1/3 by cycles (cached per process)
    causal_adaptive_norm  per-channel first-order causal adaptive normalisation (drift reference pathway, S2.1-B)
    compute_channel_mask  SNR of the calibration segments of all datasets → global top-K channels (S2.1-C)
    load_fold          the four 120-D pathways [rawz | ccnz | diff16 | diff64] concatenated to 480-D frame features (S2.1-A/D)
    support_indices    calibration frames grouped by 0.5° angle bin × motion direction, 4 frames kept per group as anchors (S2.1-E)
No intermediate data file is written.
"""
import glob
import os

import numpy as np

from . import config as C

# ====================================================================== 原始数据 → 三段切分 / raw data → three segments
def cycle_split(n_cycles, evaluated):
    """按周期数分配标定 / 验证 / 测试周期数。被测集三等分；辅助集只分标定与其余。 / Cycles per calibration / validation / test segment: thirds for evaluated datasets, calibration + rest for auxiliary ones."""
    if not evaluated:
        n_sup = min(max(1, round(C.SUP_PCT * n_cycles)), n_cycles)
        return n_sup, 0, n_cycles - n_sup
    n_sup = max(1, round(C.SUP_PCT * n_cycles))
    n_val = max(1, round(C.VAL_PCT * n_cycles))
    if n_sup + n_val >= n_cycles:
        n_sup, n_val = 1, 1
    return n_sup, n_val, n_cycles - n_sup - n_val


def supplementary_dir(data_dir=C.DATA_DIR):
    """与被测数据目录并列的辅助数据目录（默认 Supplementary/）。 / The auxiliary data folder next to the evaluated one (default Supplementary/)."""
    if os.path.abspath(data_dir) == os.path.abspath(C.DATA_DIR):
        return C.SUPP_DIR
    return os.path.join(os.path.dirname(os.path.abspath(data_dir)), os.path.basename(C.SUPP_DIR))


def find_parquet(data_dir, name):
    """定位 <name>.parquet：被测数据集在 data_dir 下，辅助数据集在并列的 Supplementary 目录下（任意子目录）。 / Locate <name>.parquet: evaluated datasets under data_dir, auxiliary ones under the neighbouring Supplementary folder (any subfolder)."""
    root = supplementary_dir(data_dir) if name in C.HOLDOUT else data_dir
    hits = glob.glob(os.path.join(root, "**", name + ".parquet"), recursive=True)
    if len(hits) != 1:
        raise FileNotFoundError("expected exactly one %s.parquet under %s, found %d" % (name, root, len(hits)))
    return hits[0]


_RAW_CACHE = {}


def load_raw(name, data_dir=C.DATA_DIR):
    """读取一个数据集并切分（进程内缓存）。

    返回 dict：x_clean[N,120]（float32 原始电阻）、y[N]（float32 角度）、
    s0,s1（标定段闭区间）、q0（验证段起点）、t0（测试段起点）、ts0,ts1（最终评测用的鲜锚段 = 验证段）、
    n_cycles（扫描周期数）、path（parquet 路径）。辅助集没有独立验证 / 测试段：q0 = t0，鲜锚段 = 标定段。
    段边界落在扫描周期边界上。

    Read one dataset and split it (cached per process). Returns a dict: x_clean[N,120] (float32 raw resistance), y[N] (float32 angle),
    s0,s1 (closed calibration interval), q0 (validation start), t0 (test start), ts0,ts1 (fresh-anchor segment = validation segment),
    n_cycles, path. Auxiliary datasets have no validation / test segment: q0 = t0 and the fresh-anchor segment is the calibration
    segment. Segment borders fall on scan-cycle borders.
    """
    key = (data_dir, name)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]
    import pandas as pd
    path = find_parquet(data_dir, name)
    df = pd.read_parquet(path)
    seg = df[C.SEGMENT_COLUMN].to_numpy()
    segvals = sorted(set(seg.tolist()))
    evaluated = name in C.FOLDS
    n_sup, n_val, _ = cycle_split(len(segvals), evaluated)
    first = lambda k: int(np.where(seg == segvals[k])[0][0])
    last = lambda k: int(np.where(seg == segvals[k])[0][-1])
    s0, s1 = first(0), last(n_sup - 1)
    q0 = s1 + 1
    if evaluated and n_val > 0:
        t0 = first(n_sup + n_val)
        ts0, ts1 = q0, t0 - 1
    else:
        t0, ts0, ts1 = q0, s0, s1
    x_clean = df[C.RAW_COLUMNS].to_numpy(np.float64).astype(np.float32)
    y = df[C.ANGLE_COLUMN].to_numpy(np.float64).astype(np.float32)
    _RAW_CACHE[key] = dict(x_clean=x_clean, y=y, s0=s0, s1=s1, q0=q0, t0=t0, ts0=ts0, ts1=ts1, fold=name,
                           n_cycles=len(segvals), n_sup=n_sup, n_val=n_val, path=path)
    return _RAW_CACHE[key]


# ====================================================================== 因果自适应归一化（drift 参考通路）/ causal adaptive normalisation (drift reference pathway)
def _iqr_cols(a):
    """逐通道四分位距；退化通道以 1.0 兜底。 / Per-channel interquartile range, 1.0 for degenerate channels."""
    q = np.nanpercentile(a, 75, axis=0) - np.nanpercentile(a, 25, axis=0)
    return np.where(np.isfinite(q) & (q > 1e-8), q, 1.0)


def causal_adaptive_norm(x0, s0, s1, H=C.CCN_HALFLIFE_MULT, cap=5.0, db_k=3.0):
    """每通道独立的一阶因果自适应归一化。

    标定段的中位数 / IQR 给出参考中心与尺度；标定段前后两半 IQR 的相对差决定尺度死区宽度；
    漂移中心与平均绝对偏差以半衰期 H × 标定段长度的 EMA 递推更新，中心被夹在 ±cap×IQR 内，
    有效尺度在死区内冻结、越界按比例跟随并钳制在 [0.5, 2]×IQR。第 t 帧输出只依赖 ≤t 帧。

    Per-channel first-order causal adaptive normalisation. Median / IQR of the calibration segment give the reference centre
    and scale; the relative IQR difference between the two halves of the calibration segment sets the dead-band width of the
    scale; the drift centre and the mean absolute deviation are updated by an EMA with half-life H × calibration length, the
    centre is clipped to ±cap×IQR, the effective scale is frozen inside the dead band, follows proportionally outside it and
    is clamped to [0.5, 2]×IQR. The output at frame t depends on frames ≤ t only.
    """
    sx = x0[s0:s1 + 1].astype(np.float64)
    slen = s1 - s0 + 1
    c0 = np.nanmedian(sx, axis=0)
    iqr0 = _iqr_cols(sx)
    half = max(slen // 2, 1)
    wob = np.abs(_iqr_cols(sx[:half]) - _iqr_cols(sx[half:])) / iqr0
    db = np.clip(db_k * wob, 0.25, 1.0)
    lo_db, hi_db = 1.0 / (1.0 + db), 1.0 + db
    a = 1.0 - 2.0 ** (-1.0 / max(H * slen, 1.0))
    lo, hi = c0 - cap * iqr0, c0 + cap * iqr0
    c = c0.copy()
    ad = iqr0 * 0.591
    out = np.empty_like(x0, dtype=np.float64)
    for i in range(len(x0)):
        row = x0[i].astype(np.float64)
        cc = np.clip(c, lo, hi)
        run = ad * 1.692
        ratio = run / iqr0
        eff = iqr0.copy()
        hiM = ratio > hi_db
        loM = ratio < lo_db
        eff[hiM] = iqr0[hiM] * (ratio[hiM] / hi_db[hiM])
        eff[loM] = iqr0[loM] * (ratio[loM] / lo_db[loM])
        eff = np.clip(eff, 0.5 * iqr0, 2.0 * iqr0)
        out[i] = (row - cc) / eff
        rn = np.nan_to_num(row, nan=0.0)
        c = (1.0 - a) * c + a * rn
        ad = (1.0 - a) * ad + a * np.abs(rn - cc)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


_CCN_CACHE = {}


def get_ccn(name, data_dir=C.DATA_DIR):
    """某折原始信号的因果归一化参考（进程内缓存）。逐通道独立，翻转变体由调用方对列做置换。 / Causal normalisation reference of one dataset (cached per process); per channel, so flipped copies permute the columns."""
    key = (data_dir, name)
    if key not in _CCN_CACHE:
        z = load_raw(name, data_dir)
        _CCN_CACHE[key] = causal_adaptive_norm(z["x_clean"].astype(np.float64), int(z["s0"]), int(z["s1"]))
    return _CCN_CACHE[key]


# ====================================================================== 通道信噪比与全局掩膜 / channel SNR and global mask
def channel_snr(x, y, s0, s1, bin_deg=C.SNR_BIN):
    """标定段逐通道信噪比：按运动方向分支、按 1° 角度分箱；
    信号 = 各箱中位数的跨箱标准差，噪声 = 箱内标准差的平均值。
    Per-channel SNR of the calibration segment: split by motion direction and 1° angle bins;
    signal = std of the bin medians across bins, noise = mean of the within-bin std."""
    m = np.arange(s0, s1 + 1)
    a = y[m]
    xs = x[m]
    bdir = np.sign(np.diff(y, prepend=y[0]))[m]
    sig_bins, noi_bins = [], []
    for b in (1.0, -1.0):
        mm = bdir == b
        if mm.sum() < 20:
            continue
        aa = a[mm]
        xx = xs[mm]
        for e in np.arange(aa.min(), aa.max() + bin_deg, bin_deg):
            sel = (aa >= e) & (aa < e + bin_deg)
            if sel.sum() >= 3:
                sig_bins.append(np.median(xx[sel], 0))
                noi_bins.append(np.std(xx[sel], 0))
    return np.std(np.array(sig_bins), 0) / (np.mean(np.array(noi_bins), 0) + 1e-9)


def compute_channel_mask(names, data_dir=C.DATA_DIR, k=C.MASK_K):
    """全部在册数据集标定段信噪比的跨集中位数 → 保留通道号（升序）。掩膜全局唯一，不含被测折身份。 / Median SNR over all catalogued datasets → kept channels (sorted); one global mask, independent of the evaluated dataset."""
    names = sorted(set(names))
    scores = []
    for n in names:
        z = load_raw(n, data_dir)
        scores.append(channel_snr(get_ccn(n, data_dir).astype(np.float64), z["y"].astype(np.float64), int(z["s0"]), int(z["s1"])))
    g = np.median(np.array(scores), 0)
    return np.sort(np.argsort(-g)[:k])


# ====================================================================== 电极环对称置换 / electrode-ring symmetry permutations
def build_flip_perms():
    """16 电极环的上下镜像 / 左右镜像 / 旋转 180° 对应的 120 维通道置换表。 / 120-channel permutations of the up-down mirror, left-right mirror and 180° rotation of the 16-electrode ring."""
    E = C.N_ELECTRODE
    table = -np.ones((E, E), np.int64)
    c = 0
    for i in range(E):
        for j in range(i + 1, E):
            table[i, j] = table[j, i] = c
            c += 1
    ud, lr, both = [], [], []
    for i in range(E):
        for j in range(i + 1, E):
            ud.append(table[E - 1 - i, E - 1 - j])
            oi = (E // 2 - 1 - i) if i <= E // 2 - 1 else int(E * 1.5 - 1 - i)
            oj = (E // 2 - 1 - j) if j <= E // 2 - 1 else int(E * 1.5 - 1 - j)
            lr.append(table[oi, oj])
            qi = i + E // 2 if i <= E // 2 - 1 else i - E // 2
            qj = j + E // 2 if j <= E // 2 - 1 else j - E // 2
            both.append(table[qi, qj])
    return {"UD": np.array(ud, np.int64), "LR": np.array(lr, np.int64), "180": np.array(both, np.int64)}


FLIP_PERMS = build_flip_perms()


def electrode_channels(e):
    """含电极 e 的 15 个通道号（升序）。 / The 15 channels that involve electrode e (sorted)."""
    return sorted(C.channel_index(i, e) for i in range(C.N_ELECTRODE) if i != e)


# ====================================================================== 帧特征 / frame features
def ema_smooth(x, half):
    """因果指数滑动平均，半衰期 half 帧。 / Causal exponential moving average with half-life `half` frames."""
    a = 1.0 - 2.0 ** (-1.0 / half)
    out = np.empty_like(x)
    m = x[0].copy()
    for t in range(len(x)):
        m = (1 - a) * m + a * x[t]
        out[t] = m
    return out


def whiten_by_support(x, s0, s1):
    """用标定段中位数 / IQR 做稳健白化并截断到 ±CLIP_Z。 / Robust whitening with the calibration-segment median / IQR, clipped to ±CLIP_Z."""
    sup = x[s0:s1 + 1]
    med = np.median(sup, 0)
    iqr = (np.percentile(sup, 75, 0) - np.percentile(sup, 25, 0)) + 1e-6
    return np.clip((x - med) / iqr, -C.CLIP_Z, C.CLIP_Z).astype(np.float32)


def feature_dim(channels=None):
    """帧特征维数：4 × 通道数。 / Frame-feature size: 4 × number of channels."""
    return 4 * (C.N_CHANNEL if channels is None else len(channels))


def build_features(x_raw, x_ccn, s0, s1, goodch, channels=None, ablation=None):
    """由原始信号与因果参考构造帧特征 [rawz | ccnz | diff16 | diff64]。

    channels 为 None 时使用全部 120 通道并按 goodch 掩膜把其余通道置零；
    给定通道子集时只保留这些列且不施加掩膜。ablation 可为 'diff0'（差分通路置零）或 'ccn0'（参考通路置零）。

    Build the frame features [rawz | ccnz | diff16 | diff64] from the raw signal and the causal reference. With channels=None
    all 120 channels are used and those outside the goodch mask are zeroed; with a channel subset only those columns are kept
    and no mask is applied. ablation may be 'diff0' (differential pathways zeroed) or 'ccn0' (reference pathway zeroed).
    """
    if channels is not None:
        keep = np.asarray(channels, np.int64)
        x_ccn = x_ccn[:, keep]
        x_raw = x_raw[:, keep]
    ccnz = whiten_by_support(x_ccn, s0, s1)
    rawz = whiten_by_support(x_raw, s0, s1)
    if channels is None:
        bad = np.ones(C.N_CHANNEL, bool)
        bad[goodch] = False
        rawz[:, bad] = 0.0
        ccnz[:, bad] = 0.0
    cols = [rawz, ccnz]
    for tau in C.DIFF_TAUS:
        cols.append((rawz - ema_smooth(rawz, tau)).astype(np.float32))
    X = np.concatenate(cols, 1).astype(np.float32)
    n = X.shape[1] // 4
    if ablation == "diff0":
        X[:, 2 * n:4 * n] = 0.0
    elif ablation == "ccn0":
        X[:, n:2 * n] = 0.0
    return X


def load_fold(name, goodch, data_dir=C.DATA_DIR, flip=None, channels=None, ablation=None,
              x_raw=None, x_ccn=None, y=None):
    """读取一个折并构造帧特征。

    返回 dict：x[N,D] 特征、y[N] 角度、s0/s1 标定段、v0 验证段起点、t0 测试段起点、ts0/ts1 鲜锚段、N。
    flip 为 'UD' / 'LR' / '180' 时在原始信号上做通道置换（角度标签不变）。
    x_raw / x_ccn / y 可覆盖数据文件中的信号（供置换重要性检验注入扰动）；只给 x_raw 时因果参考按扰动后信号重算。

    Load one dataset and build its frame features. Returns a dict: x[N,D] features, y[N] angles, s0/s1 calibration segment,
    v0 validation start, t0 test start, ts0/ts1 fresh-anchor segment, N. flip = 'UD' / 'LR' / '180' permutes the channels of the
    raw signal (labels unchanged). x_raw / x_ccn / y may override the signals of the data file (perturbations for the permutation
    importance); when only x_raw is given the causal reference is recomputed from the perturbed signal.
    """
    z = load_raw(name, data_dir)
    y = z["y"].astype(np.float32) if y is None else y
    s0, s1, v0 = int(z["s0"]), int(z["s1"]), int(z["q0"])
    t0 = int(z["t0"])
    ts0, ts1 = int(z["ts0"]), int(z["ts1"])
    if not (s1 < v0 <= t0):
        raise ValueError("segment order violated in %s: s1=%d v0=%d t0=%d" % (name, s1, v0, t0))
    if x_raw is None:
        x_raw = z["x_clean"].astype(np.float32)
        x_ccn = get_ccn(name, data_dir) if x_ccn is None else x_ccn
    elif x_ccn is None:
        x_ccn = causal_adaptive_norm(x_raw.astype(np.float64), s0, s1)
    if flip is not None:
        p = FLIP_PERMS[flip]
        x_ccn = x_ccn[:, p]
        x_raw = x_raw[:, p]
    if channels is not None:
        keep = np.asarray(channels, np.int64)
        drift_src = x_raw[:, keep]
    else:
        drift_src = x_raw
    N = len(y)
    drift_l1 = float(np.mean(np.abs(drift_src[s0:s1 + 1].mean(0) - drift_src[max(t0, N - 200):].mean(0))))
    if drift_l1 <= 0.3:
        raise ValueError("%s: raw signal drift %.3f too small; the data do not look like raw resistance recordings" % (name, drift_l1))
    X = build_features(x_raw, x_ccn, s0, s1, goodch, channels, ablation)
    return dict(x=X, y=y, s0=s0, s1=s1, v0=v0, t0=t0, ts0=ts0, ts1=ts1, N=N)


# ====================================================================== 锚抽样 / anchor sampling
def subsample_support(sidx, y, K=C.SUP_K):
    """标定帧按 (角度箱 × 运动方向) 分组，每组沿时间均匀保留 K 帧。 / Calibration frames grouped by (angle bin × motion direction), K frames kept per group, evenly in time."""
    if C.SUP_ANGSTEP <= 0 or len(sidx) <= 1:
        return sidx
    ang = np.asarray(y)[sidx].astype(np.float64)
    d = (np.sign(np.diff(ang, prepend=ang[0])) > 0).astype(np.int64)
    key = (np.floor(ang / C.SUP_ANGSTEP).astype(np.int64) * 2 + d)
    groups = {}
    for i in range(len(key)):
        groups.setdefault(int(key[i]), []).append(i)
    K = max(1, int(K))
    keep = []
    for idx in groups.values():
        if len(idx) <= K:
            keep.extend(idx)
        else:
            keep.extend(idx[j] for j in np.linspace(0, len(idx) - 1, K).astype(np.int64))
    keep.sort()
    return sidx[np.array(keep, dtype=np.int64)]


def support_indices(fd):
    """当前锚段 [s0,s1] 的锚帧索引。 / Anchor frame indices of the current anchor segment [s0,s1]."""
    return subsample_support(np.arange(fd["s0"], fd["s1"] + 1), fd["y"], C.SUP_K)


def fresh_anchor(fd):
    """把锚段切换为鲜锚段 [ts0,ts1]（最终评测口径）。 / Switch the anchor segment to the fresh one [ts0,ts1] (final-test setting)."""
    out = dict(fd)
    out["s0"], out["s1"] = fd["ts0"], fd["ts1"]
    return out


def to_frames(x, idx):
    """取若干帧并整理成模型输入形状 [B,1,D]。 / Gather frames into the model input shape [B,1,D]."""
    return np.ascontiguousarray(x[idx][:, None, :])
