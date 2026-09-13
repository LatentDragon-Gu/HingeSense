# -*- coding: utf-8 -*-
"""16 种对比方法，与主方法共享数据切分、特征与训练源池。

方法：
    ridge / rf                     帧局部：全局岭回归、随机森林
    kalman / es                    以岭回归输出为观测的常速卡尔曼滤波 / 指数平滑
    mlp / gru / tcn / transformer  帧局部 MLP；64 帧窗口的 GRU / TCN / Transformer
    kalmannet                      可学习递归滤波
    gru_aug / tcn_aug / rf_aug     训练时加入与主方法相同的三种扰动
    calib_ridge                    只用被测集自身标定段拟合的岭回归
    calib_kalman / calib_es        以 calib_ridge 输出为观测的滤波
    rawknn                         无学习的 soft-kNN（直接在 480 维特征空间匹配锚）
评测：每个方法给出顺序测试 MAE（mae_seq）与块乱序 MAE（mae_shuf，块长 96/128 × 2 个种子的平均）。
单折工作进程，由根目录 train.py --method <方法> 调度，也可直接运行：
    python -m hingesense.baselines --method gru --fold Data_01 --out results/raw/baselines/gru
产物：<out>/<fold>.json。kalman / es 以 ridge 的全序列预测为观测，复用 <out 的上级目录>/ridge/<fold>_pred.npy 缓存（无缓存时重新拟合，结果相同）。

The 16 comparison methods, sharing the data split, the features and the training pool of the decoder:
    ridge / rf                     frame-local: global ridge regression, random forest
    kalman / es                    constant-velocity Kalman filter / exponential smoothing of the ridge output
    mlp / gru / tcn / transformer  frame-local MLP; GRU / TCN / Transformer on 64-frame windows
    kalmannet                      learned recursive filter
    gru_aug / tcn_aug / rf_aug     trained with the three perturbations of the decoder
    calib_ridge                    ridge fitted on the calibration segment of the evaluated dataset only
    calib_kalman / calib_es        filters of the calib_ridge output
    rawknn                         learning-free soft-kNN in the 480-D feature space
Each method reports the sequential test MAE (mae_seq) and the block-shuffled MAE (mae_shuf, blocks of 96/128 × 2 seeds, averaged).
One dataset per process, scheduled by train.py --method <name> or run directly as shown above. Output: <out>/<fold>.json.
kalman / es observe the full-sequence ridge prediction and reuse the cache <parent of out>/ridge/<fold>_pred.npy (refitted, with
the same result, when absent).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

from . import config as C
from . import dataset as ds
from .trainer import set_determinism

METHODS = C.BASELINE_METHODS
FLIP_VARIANTS = (None, "UD", "LR", "180")
RIDGE_CAP = 1_500_000
RF_CAP = 300_000
KAL_QGRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
KAL_RGRID = (1.0, 4.0, 16.0, 64.0)
ES_HGRID = (1, 2, 4, 8, 16, 32)
WIN = 64
QBATCH = C.QBATCH
EPOCH_STEPS = C.EPOCH_STEPS
NN_LR = 1e-3
NN_WD = C.WD
CLIP_NORM = C.CLIP_NORM
EVALB = 512
BLOCKS = C.SHUFFLE_BLOCKS
KSEEDS = C.SHUFFLE_SEED_LABELS
KN_SEQLEN = 128
KN_BATCH = 64
KN_HIDDEN = 32
KN_ENCH = 64
KN_ZCHUNK = 8192
RK_TAUGRID = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)
RK_QCHUNK = 4096


class Context:
    """运行上下文：通道掩膜、数据目录、设备、训练轮数。 / Run context: channel mask, data folder, device, epochs."""

    def __init__(self, goodch, data_dir, dev, epochs, patience, log):
        self.goodch, self.data_dir, self.dev, self.epochs, self.patience, self.log = goodch, data_dir, dev, epochs, patience, log

    def load_fold(self, name, flip=None):
        return ds.load_fold(name, self.goodch, self.data_dir, flip=flip)


# ====================================================================== 训练源池与乱序置换 / training pool and shuffle permutations
def build_pool(ctx, target, cond=""):
    """LODO 训练源：(被测集 + 辅助集) 去掉被测折，按物理条件过滤，每个源折展开四个翻转变体。 / Training sources: (evaluated + auxiliary) minus the evaluated dataset, same mounting condition, four flip variants each."""
    pool_names = list(dict.fromkeys(C.FOLDS + C.HOLDOUT))
    cond = cond or C.cond_of(target)
    srcs = [f for f in pool_names if f != target and C.cond_of(f) == cond]
    if not srcs:
        raise RuntimeError("%s: no training source for cond=%s" % (target, C.COND_NAME.get(cond, cond)))
    return [(f, fl) for f in srcs for fl in FLIP_VARIANTS], cond


def block_perm(N, t0, block, klabel):
    """测试段块内乱序置换：只置换 [t0,N)，块内打乱、块间保序。种子 = SEED + 100003×klabel + 31。 / Block-shuffle permutation of the test segment [t0,N): frames shuffled within blocks, block order kept; seed = SEED + 100003×klabel + 31."""
    seed = 100003 * klabel + 31
    perm = np.arange(N)
    r = np.random.RandomState(C.SEED + seed)
    for a in range(t0, N, block):
        b = min(a + block, N)
        seg = np.arange(a, b)
        r.shuffle(seg)
        perm[a:b] = seg
    return perm


def framelocal_shuffle_blocks(preds_test, y, t0, N):
    """帧局部方法的乱序结果：顺序预测按置换重排（与重新推理等价）。 / Block-shuffled results of frame-local methods: the sequential predictions permuted (identical to re-inference)."""
    per = {}
    for bs in BLOCKS:
        for k in KSEEDS:
            perm = block_perm(N, t0, bs, k)
            pp = preds_test[perm[t0:] - t0]
            per["b%d_%d" % (bs, k)] = float(np.mean(np.abs(pp - y[perm[t0:]])))
    return per


# ====================================================================== 帧局部：ridge / rf / frame-local: ridge / rf
def collect_train_frames(ctx, pool, cap):
    """训练样本 = 各源折变体的测试区间 [t0,N) 帧 + 标定段帧；超过 cap 时按固定种子均匀抽样。 / Training frames = test interval [t0,N) + calibration frames of every source variant; subsampled with a fixed seed above cap."""
    xs, ys = [], []
    for name, flip in pool:
        fd = ctx.load_fold(name, flip)
        idx = np.concatenate([np.arange(fd["t0"], fd["N"]), np.arange(fd["s0"], fd["s1"] + 1)])
        xs.append(fd["x"][idx])
        ys.append(fd["y"][idx])
        del fd
    X = np.concatenate(xs, 0)
    yv = np.concatenate(ys, 0)
    n_total = len(X)
    if n_total > cap:
        sel = np.sort(np.random.RandomState(C.SEED).choice(n_total, cap, replace=False))
        X, yv = X[sel], yv[sel]
    return X, yv, n_total


def fit_ridge_predict_full(ctx, target, fd_t, cond):
    from sklearn.linear_model import Ridge
    pool, cond = build_pool(ctx, target, cond)
    X, yv, n_total = collect_train_frames(ctx, pool, RIDGE_CAP)
    ctx.log("  [%s] ridge cond=%s sources=%d frames=%d/%d" % (target, C.COND_NAME.get(cond, cond), len(pool), len(X), n_total))
    mdl = Ridge(alpha=1.0)
    mdl.fit(X, yv)
    return mdl.predict(fd_t["x"]).astype(np.float64), dict(n_train=int(len(X)), n_total=int(n_total), n_sources=len(pool))


def ridge_cache_path(out_dir, target):
    """ridge 全序列预测缓存：<各方法目录的上级>/ridge/<fold>_pred.npy。 / Cache of the full-sequence ridge prediction: <parent of the method folders>/ridge/<fold>_pred.npy."""
    return os.path.join(os.path.dirname(os.path.normpath(out_dir)), "ridge", target + "_pred.npy")


def ridge_full_preds(ctx, target, fd_t, out_dir, cond):
    """kalman / es 的观测源：复用同折 ridge 全序列预测缓存。 / Observation source of kalman / es: the cached full-sequence ridge prediction of the same dataset."""
    cpath = ridge_cache_path(out_dir, target)
    if os.path.exists(cpath):
        p = np.load(cpath)
        if len(p) == fd_t["N"]:
            return p.astype(np.float64), {"ridge_cached": True}
    preds, extra = fit_ridge_predict_full(ctx, target, fd_t, cond)
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    np.save(cpath, preds.astype(np.float32))
    return preds, extra


def run_ridge(ctx, target, fd, out_dir, cond):
    preds_full, extra = fit_ridge_predict_full(ctx, target, fd, cond)
    cpath = ridge_cache_path(out_dir, target)
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    np.save(cpath, preds_full.astype(np.float32))
    t0, N = fd["t0"], fd["N"]
    y = fd["y"].astype(np.float64)
    mae_seq = float(np.mean(np.abs(preds_full[t0:] - y[t0:])))
    return mae_seq, framelocal_shuffle_blocks(preds_full[t0:], y, t0, N), extra


def run_rf(ctx, target, fd, out_dir, cond):
    from sklearn.ensemble import RandomForestRegressor
    pool, cond = build_pool(ctx, target, cond)
    X, yv, n_total = collect_train_frames(ctx, pool, RF_CAP)
    ctx.log("  [%s] rf cond=%s sources=%d frames=%d/%d" % (target, C.COND_NAME.get(cond, cond), len(pool), len(X), n_total))
    mdl = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=32, random_state=C.SEED)
    mdl.fit(X, yv)
    t0, N = fd["t0"], fd["N"]
    y = fd["y"].astype(np.float64)
    preds_test = mdl.predict(fd["x"][t0:N]).astype(np.float64)
    mae_seq = float(np.mean(np.abs(preds_test - y[t0:])))
    return mae_seq, framelocal_shuffle_blocks(preds_test, y, t0, N), dict(n_train=int(len(X)), n_total=int(n_total), n_sources=len(pool))


# ====================================================================== 滤波：kalman / es / filters: kalman / es
def kf_filter(obs, q, r, state=None):
    """常速卡尔曼滤波：状态 [角度, 角速度]，dt=1，Q = q·[[1/3,1/2],[1/2,1]]，R = r。 / Constant-velocity Kalman filter: state [angle, rate], dt=1, Q = q·[[1/3,1/2],[1/2,1]], R = r."""
    q00, q01, q11 = q / 3.0, q / 2.0, float(q)
    if state is None:
        a, v = float(obs[0]), 0.0
        p00, p01, p11 = 1e3, 0.0, 1e3
    else:
        a, v, p00, p01, p11 = state
    ests = np.empty(len(obs), np.float64)
    for i in range(len(obs)):
        a = a + v
        n00 = p00 + 2.0 * p01 + p11 + q00
        n01 = p01 + p11 + q01
        n11 = p11 + q11
        s = n00 + r
        k0 = n00 / s
        k1 = n01 / s
        resid = float(obs[i]) - a
        a += k0 * resid
        v += k1 * resid
        p00 = (1.0 - k0) * n00
        p01 = (1.0 - k0) * n01
        p11 = n11 - k1 * n01
        ests[i] = a
    return ests, (a, v, p00, p01, p11)


def ema_run(obs, half, m=None):
    """因果指数平滑（半衰期 half 帧），可从给定状态延续。 / Causal exponential smoothing (half-life `half` frames), optionally continued from a state."""
    a = 1.0 - 2.0 ** (-1.0 / half)
    out = np.empty(len(obs), np.float64)
    if m is None:
        m = float(obs[0])
    for i in range(len(obs)):
        m = (1.0 - a) * m + a * float(obs[i])
        out[i] = m
    return out, m


def _run_filter(ctx, target, fd, obs, grid, warm_fn, name):
    """滤波基线公共流程：验证段网格选优（从 s0 热身）→ 顺序测试 → 乱序观测流重滤（状态延续自热身末态）。 / Shared filter procedure: grid search on the validation segment (warm-up from s0) → sequential test → re-filtering of the shuffled observation stream from the warm-up state."""
    s0, v0, t0, N = fd["s0"], fd["v0"], fd["t0"], fd["N"]
    y = fd["y"].astype(np.float64)
    best = None
    for hp in grid:
        ests, _ = warm_fn(obs[s0:t0], hp)
        vmae = float(np.mean(np.abs(ests[v0 - s0:] - y[v0:t0])))
        if best is None or vmae < best[0] - 1e-12:
            best = (vmae, hp)
    vmae, hp = best
    ctx.log("  [%s] %s hp=%s val_mae=%.4f" % (target, name, hp, vmae))
    _, state = warm_fn(obs[s0:t0], hp)
    ests_seq, _ = warm_fn(obs[t0:N], hp, state)
    mae_seq = float(np.mean(np.abs(ests_seq - y[t0:N])))
    per = {}
    for bs in BLOCKS:
        for k in KSEEDS:
            perm = block_perm(N, t0, bs, k)
            ests_p, _ = warm_fn(obs[perm[t0:]], hp, state)
            per["b%d_%d" % (bs, k)] = float(np.mean(np.abs(ests_p - y[perm[t0:]])))
    return mae_seq, per, vmae, hp


def _kal_warm(obs_seg, hp, state=None):
    return kf_filter(obs_seg, hp[0], hp[1], state)


def _es_warm(obs_seg, hp, state=None):
    return ema_run(obs_seg, hp, state)


def run_kalman(ctx, target, fd, out_dir, cond):
    obs, rextra = ridge_full_preds(ctx, target, fd, out_dir, cond)
    mae_seq, per, vmae, hp = _run_filter(ctx, target, fd, obs, [(q, r) for q in KAL_QGRID for r in KAL_RGRID], _kal_warm, "kalman")
    return mae_seq, per, dict(q=hp[0], r=hp[1], val_mae=round(vmae, 4), **rextra)


def run_es(ctx, target, fd, out_dir, cond):
    obs, rextra = ridge_full_preds(ctx, target, fd, out_dir, cond)
    mae_seq, per, vmae, hp = _run_filter(ctx, target, fd, obs, list(ES_HGRID), _es_warm, "es")
    return mae_seq, per, dict(h=int(hp), val_mae=round(vmae, 4), **rextra)


def calib_ridge_full_preds(fd):
    """只用被测集标定段拟合的岭回归，对全序列逐帧预测。 / Ridge fitted on the calibration segment only, predicting the whole sequence."""
    from sklearn.linear_model import Ridge
    s0, s1 = fd["s0"], fd["s1"]
    mdl = Ridge(alpha=1.0)
    mdl.fit(fd["x"][s0:s1 + 1], fd["y"][s0:s1 + 1])
    return mdl.predict(fd["x"]).astype(np.float64), dict(n_calib=int(s1 - s0 + 1))


def run_calib_ridge(ctx, target, fd, out_dir, cond):
    from sklearn.linear_model import Ridge
    s0, s1, t0, N = fd["s0"], fd["s1"], fd["t0"], fd["N"]
    mdl = Ridge(alpha=1.0)
    mdl.fit(fd["x"][s0:s1 + 1], fd["y"][s0:s1 + 1])
    y = fd["y"].astype(np.float64)
    preds_test = mdl.predict(fd["x"][t0:N]).astype(np.float64)
    extra = dict(n_calib=int(s1 - s0 + 1))
    ctx.log("  [%s] calib_ridge n_calib=%d" % (target, extra["n_calib"]))
    mae_seq = float(np.mean(np.abs(preds_test - y[t0:])))
    return mae_seq, framelocal_shuffle_blocks(preds_test, y, t0, N), extra


def run_calib_kalman(ctx, target, fd, out_dir, cond):
    obs, rextra = calib_ridge_full_preds(fd)
    mae_seq, per, vmae, hp = _run_filter(ctx, target, fd, obs, [(q, r) for q in KAL_QGRID for r in KAL_RGRID], _kal_warm, "calib_kalman")
    return mae_seq, per, dict(q=hp[0], r=hp[1], val_mae=round(vmae, 4), **rextra)


def run_calib_es(ctx, target, fd, out_dir, cond):
    obs, rextra = calib_ridge_full_preds(fd)
    mae_seq, per, vmae, hp = _run_filter(ctx, target, fd, obs, list(ES_HGRID), _es_warm, "calib_es")
    return mae_seq, per, dict(h=int(hp), val_mae=round(vmae, 4), **rextra)


# ====================================================================== 神经网络基线 / neural-network baselines
class GRUReg(nn.Module):
    """[B,64,480] → 2 层 GRU(96) → 末步隐状态 → 角度。 / [B,64,480] → 2-layer GRU(96) → last hidden state → angle."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(4 * C.N_CHANNEL, 96, num_layers=2, batch_first=True)
        self.head = nn.Linear(96, 1)

    def forward(self, xw):
        out, _ = self.gru(xw)
        return self.head(out[:, -1, :]).squeeze(-1)


class TCNReg(nn.Module):
    """[B,64,480] → 4 层因果空洞卷积（128 通道，核 3，空洞 1/2/4/8）→ 末帧 → 角度。 / [B,64,480] → 4 causal dilated convolutions (128 channels, kernel 3, dilation 1/2/4/8) → last frame → angle."""

    def __init__(self):
        super().__init__()
        chans = [4 * C.N_CHANNEL, 128, 128, 128, 128]
        dils = [1, 2, 4, 8]
        self.convs = nn.ModuleList([nn.Conv1d(chans[i], chans[i + 1], 3, dilation=dils[i]) for i in range(4)])
        self.act = nn.GELU()
        self.head = nn.Linear(128, 1)

    def forward(self, xw):
        h = xw.transpose(1, 2)
        for conv in self.convs:
            d = conv.dilation[0]
            h = self.act(conv(nn.functional.pad(h, (2 * d, 0))))
        return self.head(h[:, :, -1]).squeeze(-1)


class TransformerReg(nn.Module):
    """[B,64,480] → 线性投影 96 + 可学习位置编码 → 2 层 TransformerEncoder → 末帧 token → 角度。 / [B,64,480] → linear projection to 96 + learned positions → 2-layer TransformerEncoder → last token → angle."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4 * C.N_CHANNEL, 96)
        self.pos = nn.Parameter(torch.zeros(WIN, 96))
        layer = nn.TransformerEncoderLayer(d_model=96, nhead=4, dim_feedforward=256, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(96, 1)

    def forward(self, xw):
        h = self.proj(xw) + self.pos.unsqueeze(0)
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


class MLPReg(nn.Module):
    """帧局部 MLP：480 → 256 → 256 → 1。 / Frame-local MLP: 480 → 256 → 256 → 1."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * C.N_CHANNEL, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, xb):
        return self.net(xb).squeeze(-1)


class KalmanNetReg(nn.Module):
    """可学习递归滤波：观测编码器给出伪观测 z_t，状态 [角度, 角速度] 常速预测，
    GRUCell(输入 [创新, z_t, 预测角度]) → 两维增益 → 状态更新。
    Learned recursive filter: an observation encoder gives a pseudo-observation z_t, the state [angle, rate] is predicted at
    constant velocity, a GRUCell (input [innovation, z_t, predicted angle]) gives a two-dimensional gain → state update."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4 * C.N_CHANNEL, KN_ENCH), nn.GELU(), nn.Linear(KN_ENCH, 1))
        self.cell = nn.GRUCell(3, KN_HIDDEN)
        self.kout = nn.Linear(KN_HIDDEN, 2)

    def encode(self, x):
        return self.enc(x).squeeze(-1)

    def filter_z(self, z, state=None):
        B, T = z.shape
        if state is None:
            a = z[:, 0]
            v = torch.zeros(B, device=z.device, dtype=z.dtype)
            h = torch.zeros(B, KN_HIDDEN, device=z.device, dtype=z.dtype)
        else:
            a, v, h = state
        outs = []
        for t in range(T):
            a = a + v
            zt = z[:, t]
            e = zt - a
            h = self.cell(torch.stack([e, zt, a], dim=1), h)
            K = torch.tanh(self.kout(h))
            a = a + K[:, 0] * e
            v = v + K[:, 1] * e
            a = torch.clamp(a, -150.0, 150.0)
            v = torch.clamp(v, -15.0, 15.0)
            outs.append(a)
        return torch.stack(outs, dim=1), (a, v, h)


def make_windows(x, idx, base=0):
    """帧下标 → 64 帧历史窗 [B,WIN,480]（越过折首时复制首帧）。x 可从 base 起截尾存放。 / Frame indices → 64-frame history windows [B,WIN,480] (first frame repeated before the start); x may be stored from base on."""
    pos = np.asarray(idx, np.int64)[:, None] - np.arange(WIN - 1, -1, -1)[None, :]
    np.clip(pos, base, None, out=pos)
    return np.ascontiguousarray(x[pos - base])


def apply_query_aug(xw, rng, dev):
    """主方法同款三扰动作用于 query 窗的前 240 维：局部漂移噪声（窗内共享）→ 增益 → 偏置。 / The decoder's three perturbations on the first 240 dims of the query windows: local drift noise (shared within a window) → gain → offset."""
    qn = xw.shape[0]
    nch = 4 * C.N_CHANNEL
    ramp = torch.from_numpy((rng.standard_normal((qn, 1, 240)).astype(np.float32) * C.RAMP_STD)).to(dev)
    xw[:, :, :240] = xw[:, :, :240] + ramp
    gg = torch.from_numpy((1 + rng.standard_normal((1, 1, 120)).astype(np.float32) * C.GAIN_STD)).to(dev).repeat(1, 1, 2)
    g_aug = torch.ones(1, 1, nch, device=dev)
    g_aug[:, :, :240] = gg
    oo = torch.from_numpy((rng.standard_normal((1, 1, 120)).astype(np.float32) * C.OFFSET_STD)).to(dev).repeat(1, 1, 2)
    o_aug = torch.zeros(1, 1, nch, device=dev)
    o_aug[:, :, :240] = oo
    return xw * g_aug + o_aug


def _step(model, opt, loss):
    if not torch.isfinite(loss):
        opt.zero_grad()
        return
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    opt.step()


def train_step_window(model, opt, srcs, rng, dev, aug=False):
    """窗口模型单步：随机源折 → 随机 QBATCH 个 query 帧 → 建窗 →（可选扰动）→ MSE。 / One step of a window model: random source → QBATCH random query frames → windows → (optional perturbation) → MSE."""
    s = srcs[rng.randint(len(srcs))]
    qn = min(QBATCH, s["N"] - s["t0"])
    qidx = rng.randint(s["t0"], s["N"], size=qn)
    xw = torch.from_numpy(make_windows(s["x"], qidx, s["base"])).to(dev)
    if aug:
        xw = apply_query_aug(xw, rng, dev)
    qy = torch.from_numpy(s["y"][qidx].astype(np.float32)).to(dev)
    _step(model, opt, ((model(xw) - qy) ** 2).mean())


def train_step_frame(model, opt, srcs, rng, dev):
    """帧局部 MLP 单步。 / One step of the frame-local MLP."""
    s = srcs[rng.randint(len(srcs))]
    qn = min(QBATCH, s["N"] - s["t0"])
    qidx = rng.randint(s["t0"], s["N"], size=qn)
    xb = torch.from_numpy(np.ascontiguousarray(s["x"][qidx])).to(dev)
    qy = torch.from_numpy(s["y"][qidx].astype(np.float32)).to(dev)
    _step(model, opt, ((model(xb) - qy) ** 2).mean())


@torch.no_grad()
def eval_windows(model, x, y, idx, dev, base=0):
    preds = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), EVALB):
        sub = idx[i:i + EVALB]
        preds[i:i + EVALB] = model(torch.from_numpy(make_windows(x, sub, base)).to(dev)).float().cpu().numpy()
    return float(np.mean(np.abs(preds - y[idx].astype(np.float32)))), preds


@torch.no_grad()
def eval_frames(model, x, y, idx, dev):
    preds = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), EVALB):
        sub = idx[i:i + EVALB]
        preds[i:i + EVALB] = model(torch.from_numpy(np.ascontiguousarray(x[sub].astype(np.float32, copy=False))).to(dev)).float().cpu().numpy()
    return float(np.mean(np.abs(preds - y[idx].astype(np.float32)))), preds


def _fit_nn(ctx, target, fd_t, name, model, step_fn, eval_fn, srcs):
    """神经网络基线的公共训练循环：AdamW、每 epoch 200 步、验证段早停、载回最优。 / Shared training loop of the neural baselines: AdamW, 200 steps per epoch, early stopping on the validation segment, best weights restored."""
    t_start = time.time()
    opt = torch.optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=NN_WD)
    rng = np.random.RandomState(C.SEED)
    v_idx = np.arange(fd_t["v0"], fd_t["t0"])
    best_mae, best_state, best_ep, pat = float("inf"), None, -1, 0
    for ep in range(ctx.epochs):
        model.train()
        for _ in range(EPOCH_STEPS):
            step_fn(model, opt, srcs, rng)
        model.eval()
        vmae, _ = eval_fn(model, fd_t["x"], fd_t["y"], v_idx)
        if vmae < best_mae - 1e-6:
            best_mae, best_ep, pat = vmae, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        ctx.log("  [%s] %s ep %d val %.4f best %.4f@ep%d pat %d (%.0fs)" % (target, name, ep, vmae, best_mae, best_ep, pat, time.time() - t_start))
        if pat >= ctx.patience:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(ctx.dev) for k, v in best_state.items()})
    model.eval()
    return dict(best_ep=int(best_ep), best_val=round(float(best_mae), 4))


def run_window_model(ctx, target, fd_t, out_dir, cond, kind, aug=False):
    """gru / tcn / transformer 及其加扰动变体。乱序评测在重排后的序列上重建窗口真实推理。 / gru / tcn / transformer and their perturbed variants; the shuffled evaluation rebuilds the windows on the permuted sequence."""
    pool, cond = build_pool(ctx, target, cond)
    name = kind + ("_aug" if aug else "")
    ctx.log("  [%s] %s cond=%s sources=%d" % (target, name, C.COND_NAME.get(cond, cond), len(pool)))
    srcs = []
    for fname, flip in pool:
        fd = ctx.load_fold(fname, flip)
        base = max(0, fd["t0"] - (WIN - 1))
        srcs.append(dict(x=np.ascontiguousarray(fd["x"][base:]), y=fd["y"], t0=fd["t0"], N=fd["N"], base=base))
        del fd
    torch.manual_seed(C.SEED)
    model = {"gru": GRUReg, "tcn": TCNReg, "transformer": TransformerReg}[kind]().to(ctx.dev)
    dev = ctx.dev
    extra = _fit_nn(ctx, target, fd_t, name, model,
                    lambda m, o, s, r: train_step_window(m, o, s, r, dev, aug),
                    lambda m, x, y, idx: eval_windows(m, x, y, idx, dev), srcs)
    t0, N = fd_t["t0"], fd_t["N"]
    t_idx = np.arange(t0, N)
    mae_seq, _ = eval_windows(model, fd_t["x"], fd_t["y"], t_idx, dev)
    per = {}
    for bs in BLOCKS:
        for k in KSEEDS:
            perm = block_perm(N, t0, bs, k)
            per["b%d_%d" % (bs, k)], _ = eval_windows(model, fd_t["x"][perm], fd_t["y"][perm], t_idx, dev)
    extra["n_sources"] = len(pool)
    return mae_seq, per, extra


def run_mlp(ctx, target, fd_t, out_dir, cond):
    pool, cond = build_pool(ctx, target, cond)
    ctx.log("  [%s] mlp cond=%s sources=%d" % (target, C.COND_NAME.get(cond, cond), len(pool)))
    srcs = []
    for fname, flip in pool:
        fd = ctx.load_fold(fname, flip)
        srcs.append(dict(x=np.ascontiguousarray(fd["x"]), y=fd["y"], t0=fd["t0"], N=fd["N"]))
        del fd
    torch.manual_seed(C.SEED)
    model = MLPReg().to(ctx.dev)
    dev = ctx.dev
    extra = _fit_nn(ctx, target, fd_t, "mlp", model, lambda m, o, s, r: train_step_frame(m, o, s, r, dev),
                    lambda m, x, y, idx: eval_frames(m, x, y, idx, dev), srcs)
    t0, N = fd_t["t0"], fd_t["N"]
    y = fd_t["y"].astype(np.float64)
    _, preds_test = eval_frames(model, fd_t["x"], fd_t["y"], np.arange(t0, N), dev)
    preds_test = preds_test.astype(np.float64)
    mae_seq = float(np.mean(np.abs(preds_test - y[t0:])))
    extra["n_sources"] = len(pool)
    return mae_seq, framelocal_shuffle_blocks(preds_test, y, t0, N), extra


def run_rf_aug(ctx, target, fd, out_dir, cond):
    """随机森林 + 训练帧三扰动（逐帧独立）。 / Random forest with the three perturbations applied to the training frames (independently per frame)."""
    from sklearn.ensemble import RandomForestRegressor
    pool, cond = build_pool(ctx, target, cond)
    X, yv, n_total = collect_train_frames(ctx, pool, RF_CAP)
    ctx.log("  [%s] rf_aug cond=%s sources=%d frames=%d/%d" % (target, C.COND_NAME.get(cond, cond), len(pool), len(X), n_total))
    rng = np.random.RandomState(C.SEED)
    X = X.astype(np.float32, copy=True)
    n = len(X)
    ramp = rng.standard_normal((n, 240)).astype(np.float32) * C.RAMP_STD
    gain = 1.0 + rng.standard_normal((n, 120)).astype(np.float32) * C.GAIN_STD
    gain = np.concatenate([gain, gain], axis=1)
    off = rng.standard_normal((n, 120)).astype(np.float32) * C.OFFSET_STD
    off = np.concatenate([off, off], axis=1)
    X[:, :240] = (X[:, :240] + ramp) * gain + off
    del ramp, gain, off
    mdl = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=32, random_state=C.SEED)
    mdl.fit(X, yv)
    t0, N = fd["t0"], fd["N"]
    y = fd["y"].astype(np.float64)
    preds_test = mdl.predict(fd["x"][t0:N]).astype(np.float64)
    mae_seq = float(np.mean(np.abs(preds_test - y[t0:])))
    return mae_seq, framelocal_shuffle_blocks(preds_test, y, t0, N), dict(n_train=int(n), n_total=int(n_total), n_sources=len(pool))


@torch.no_grad()
def kn_encode_stream(model, x_np, dev):
    zs = []
    for i in range(0, len(x_np), KN_ZCHUNK):
        xb = torch.from_numpy(np.ascontiguousarray(x_np[i:i + KN_ZCHUNK].astype(np.float32, copy=False))).to(dev)
        zs.append(model.encode(xb.unsqueeze(0))[0])
    return torch.cat(zs)


def kn_sample_batch(srcs, rng):
    """采 KN_BATCH 条长 KN_SEQLEN 的连续子序列（不跨折）。 / Sample KN_BATCH contiguous subsequences of length KN_SEQLEN (within one dataset)."""
    nch = 4 * C.N_CHANNEL
    xs = np.empty((KN_BATCH, KN_SEQLEN, nch), np.float32)
    ys = np.empty((KN_BATCH, KN_SEQLEN), np.float32)
    n = 0
    while n < KN_BATCH:
        s = srcs[rng.randint(len(srcs))]
        st = rng.randint(0, s["N"])
        if st + KN_SEQLEN > s["N"]:
            continue
        xs[n] = s["x"][st:st + KN_SEQLEN]
        ys[n] = s["y"][st:st + KN_SEQLEN]
        n += 1
    return xs, ys


def run_kalmannet(ctx, target, fd_t, out_dir, cond):
    pool, cond = build_pool(ctx, target, cond)
    ctx.log("  [%s] kalmannet cond=%s sources=%d" % (target, C.COND_NAME.get(cond, cond), len(pool)))
    srcs = []
    for fname, flip in pool:
        fd = ctx.load_fold(fname, flip)
        srcs.append(dict(x=np.ascontiguousarray(fd["x"].astype(np.float32, copy=False)), y=fd["y"].astype(np.float32), N=fd["N"]))
        del fd
    if not any(s["N"] >= KN_SEQLEN for s in srcs):
        raise RuntimeError("%s: no source long enough for kalmannet" % target)
    dev = ctx.dev
    torch.manual_seed(C.SEED)
    model = KalmanNetReg().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=NN_LR, weight_decay=NN_WD)
    rng = np.random.RandomState(C.SEED)
    s0, v0, t0, N = fd_t["s0"], fd_t["v0"], fd_t["t0"], fd_t["N"]
    vlo = max(0, v0 - (KN_SEQLEN - 1))
    y_val = fd_t["y"][v0:t0].astype(np.float32)
    t_start = time.time()
    best_mae, best_state, best_ep, pat = float("inf"), None, -1, 0
    for ep in range(ctx.epochs):
        model.train()
        for _ in range(EPOCH_STEPS):
            xs, ys = kn_sample_batch(srcs, rng)
            angles, _ = model.filter_z(model.encode(torch.from_numpy(xs).to(dev)))
            _step(model, opt, ((angles - torch.from_numpy(ys).to(dev)) ** 2).mean())
        model.eval()
        with torch.no_grad():
            av, _ = model.filter_z(kn_encode_stream(model, fd_t["x"][vlo:t0], dev).unsqueeze(0))
        vmae = float(np.mean(np.abs(av[0, v0 - vlo:].cpu().numpy() - y_val)))
        if vmae < best_mae - 1e-6:
            best_mae, best_ep, pat = vmae, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        ctx.log("  [%s] kalmannet ep %d val %.4f best %.4f@ep%d pat %d (%.0fs)" % (target, ep, vmae, best_mae, best_ep, pat, time.time() - t_start))
        if pat >= ctx.patience:
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    model.eval()
    y = fd_t["y"].astype(np.float64)
    labels, idx_list, perms = ["seq"], [np.arange(s0, N)], {"seq": None}
    for bs in BLOCKS:
        for k in KSEEDS:
            lab = "b%d_%d" % (bs, k)
            perm = block_perm(N, t0, bs, k)
            perms[lab] = perm
            idx_list.append(perm[s0:])
            labels.append(lab)
    with torch.no_grad():
        zs = torch.stack([kn_encode_stream(model, fd_t["x"][idx], dev) for idx in idx_list])
        angs, _ = model.filter_z(zs)
    angs = angs.cpu().numpy().astype(np.float64)
    off = t0 - s0
    mae_seq = float(np.mean(np.abs(angs[0, off:] - y[t0:N])))
    per = {lab: float(np.mean(np.abs(angs[i, off:] - y[perms[lab][t0:]]))) for i, lab in enumerate(labels[1:], start=1)}
    return mae_seq, per, dict(best_ep=int(best_ep), best_val=round(float(best_mae), 4), n_sources=len(pool))


# ====================================================================== 无学习 soft-kNN / learning-free soft-kNN
@torch.no_grad()
def rawknn_soft(x, y, sidx, qidx, tau, dev):
    xs = torch.from_numpy(np.ascontiguousarray(x[sidx].astype(np.float32, copy=False))).to(dev)
    ys = torch.from_numpy(y[sidx].astype(np.float32)).to(dev)
    preds = np.empty(len(qidx), np.float32)
    for i in range(0, len(qidx), RK_QCHUNK):
        sub = qidx[i:i + RK_QCHUNK]
        xq = torch.from_numpy(np.ascontiguousarray(x[sub].astype(np.float32, copy=False))).to(dev)
        d2 = torch.cdist(xq, xs) ** 2
        preds[i:i + RK_QCHUNK] = (torch.softmax(-d2 / tau, dim=1) @ ys).cpu().numpy()
    return preds


def run_rawknn(ctx, target, fd, out_dir, cond):
    """锚与主方法同法抽取；温度 τ 在验证段上网格选优；最终用鲜锚预测测试段。 / Anchors selected as for the decoder; temperature τ chosen on the validation segment; test segment predicted with fresh anchors."""
    v0, t0, N = fd["v0"], fd["t0"], fd["N"]
    sidx_old = ds.support_indices(fd)
    v_idx = np.arange(v0, t0)
    best = None
    for tau in RK_TAUGRID:
        pv = rawknn_soft(fd["x"], fd["y"], sidx_old, v_idx, tau, ctx.dev)
        vmae = float(np.mean(np.abs(pv.astype(np.float64) - fd["y"][v_idx].astype(np.float64))))
        if best is None or vmae < best[0] - 1e-12:
            best = (vmae, tau)
    vmae, tau = best
    sidx_fresh = ds.support_indices(ds.fresh_anchor(fd))
    ctx.log("  [%s] rawknn tau=%s val_mae=%.4f anchors=%d/%d" % (target, tau, vmae, len(sidx_old), len(sidx_fresh)))
    y = fd["y"].astype(np.float64)
    preds_test = rawknn_soft(fd["x"], fd["y"], sidx_fresh, np.arange(t0, N), tau, ctx.dev).astype(np.float64)
    mae_seq = float(np.mean(np.abs(preds_test - y[t0:])))
    return mae_seq, framelocal_shuffle_blocks(preds_test, y, t0, N), dict(tau=float(tau), n_anchor=int(len(sidx_fresh)), n_anchor_val=int(len(sidx_old)), val_mae=round(vmae, 4))


RUNNERS = {
    "ridge": run_ridge, "rf": run_rf, "kalman": run_kalman, "es": run_es, "rf_aug": run_rf_aug, "kalmannet": run_kalmannet,
    "mlp": run_mlp, "calib_ridge": run_calib_ridge, "rawknn": run_rawknn, "calib_kalman": run_calib_kalman, "calib_es": run_calib_es,
    "gru": lambda c, t, f, o, cond: run_window_model(c, t, f, o, cond, "gru"),
    "tcn": lambda c, t, f, o, cond: run_window_model(c, t, f, o, cond, "tcn"),
    "transformer": lambda c, t, f, o, cond: run_window_model(c, t, f, o, cond, "transformer"),
    "gru_aug": lambda c, t, f, o, cond: run_window_model(c, t, f, o, cond, "gru", aug=True),
    "tcn_aug": lambda c, t, f, o, cond: run_window_model(c, t, f, o, cond, "tcn", aug=True),
}


def main(argv=None):
    from .runner import machine_info, format_machine_info
    ap = argparse.ArgumentParser(description="comparison methods (one method, one fold per process)")
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--fold", required=True)
    ap.add_argument("--out", required=True, help="output directory of this method, e.g. results/raw/baselines/gru")
    ap.add_argument("--data-dir", default=C.DATA_DIR)
    ap.add_argument("--cond", default="", choices=[""] + list(C.COND_CLI), help="mounting condition of the training sources (noload / assembly); default: the condition of the evaluated dataset")
    ap.add_argument("--mask-k", type=int, default=C.MASK_K)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    args = ap.parse_args(argv)
    set_determinism(C.SEED)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def log(msg):
        print(msg, flush=True)

    log("command: " + "python -m hingesense.baselines " + " ".join(sys.argv[1:]))
    for line in format_machine_info(machine_info()):
        log(line)
    goodch = ds.compute_channel_mask(C.FOLDS + C.HOLDOUT, args.data_dir, args.mask_k)
    ctx = Context(goodch, args.data_dir, dev, args.epochs, args.patience, log)
    fd = ctx.load_fold(args.fold)
    if fd["t0"] <= fd["v0"]:
        raise RuntimeError("%s has no validation segment" % args.fold)
    log("=== %s %s dev=%s ===" % (args.method, args.fold, dev))
    mae_seq, per, extra = RUNNERS[args.method](ctx, args.fold, fd, args.out, C.COND_CLI[args.cond] if args.cond else "")
    mae_shuf = float(np.mean([per["b%d_%d" % (b, k)] for b in BLOCKS for k in KSEEDS]))
    res = dict(method=args.method, fold=args.fold, dataset_no=C.DATASET_NO.get(args.fold), mae_seq=round(float(mae_seq), 4),
               mae_shuf=round(mae_shuf, 4), per_block={k: round(float(v), 4) for k, v in per.items()}, extra=extra,
               mae_seq_exact=float(mae_seq), mae_shuf_exact=mae_shuf, seed=C.SEED, device=str(dev))
    with open(os.path.join(args.out, args.fold + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log("*** %s %s mae_seq=%s mae_shuf=%s" % (args.method, args.fold, res["mae_seq"], res["mae_shuf"]))


if __name__ == "__main__":
    main()
