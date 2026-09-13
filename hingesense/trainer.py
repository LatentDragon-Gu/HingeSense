# -*- coding: utf-8 -*-
"""留一数据集（LODO）训练与评测：单折工作进程。由根目录 train.py 调度，也可直接运行：
    python -m hingesense.trainer --target Data_01 --out results/raw/decoder
产物（均在 --out 目录，以折名命名）：<fold>.json（测试段 MAE：mae_test 为 3 位小数、mae_test_exact 为未取整值，
以及种子、早停轮次、耗时等）、<fold>.npy（测试段逐帧预测）、<fold>_model.pt（早停最优权重）。
开关：--ablation <config.ABLATIONS 中的配置名>、--channels（通道子集）、--mask-k、--seed。

协议：
    训练源 = 同物理条件的其余数据集（被测集 + 辅助集）及其三种电极对称翻转副本；
    每步从一个源折的测试区间随机取 QBATCH 帧作为 query，以该源折标定段的锚做 soft-kNN 读出，
    损失 = MSE(融合读出) + AUXW × SmoothL1(raw 臂辅助读出)；
    早停以被测集验证段（锚 = 标定段）的 MAE 为准；最终评测锚 = 鲜锚段（验证段），报告测试段 MAE。
随机数抽取顺序（query 下标 → 局部漂移噪声 → 增益 → 偏置）与各折的随机流种子决定结果的确定性，不可改动。

Leave-one-dataset-out training and evaluation, one dataset per process. Scheduled by train.py, or run directly:
    python -m hingesense.trainer --target Data_01 --out results/raw/decoder
Outputs (in --out, named after the dataset): <fold>.json (test MAE: mae_test rounded to 3 decimals, mae_test_exact unrounded,
plus seed, early-stopping epoch, time), <fold>.npy (frame-wise test predictions), <fold>_model.pt (best weights).
Switches: --ablation <name from config.ABLATIONS>, --channels (channel subset), --mask-k, --seed.
Protocol: training sources = the other datasets of the same mounting condition (evaluated + auxiliary) and their three
electrode-symmetry copies; every step draws QBATCH query frames from the test interval of one source dataset and reads them
out by soft-kNN against the anchors of that dataset's calibration segment; loss = MSE(fused read-out) + AUXW × SmoothL1(raw-arm
read-out); early stopping on the validation segment of the evaluated dataset (anchors = calibration segment); final test with
fresh anchors (validation segment), reporting the test-segment MAE. The order in which random numbers are drawn (query indices →
local drift noise → gain → offset) and the per-dataset seeds fix the results and must not be changed.
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
from .model import HingeNet, soft_readout



def set_determinism(seed):
    """全局种子与确定性算子设置。 / Global seeds and deterministic kernels."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def log(msg):
    """工作进程只写标准输出；调度它的根脚本把标准输出保存为 <fold>.log。 / Workers only print; the root script saves stdout as <fold>.log."""
    print(msg, flush=True)


def build_pool(targets, goodch, cond, data_dir, channels, ablation, with_flips):
    """构造训练数据池：{折名: 折数据}，翻转副本以 "折名@UD" 等键追加在末尾。插入顺序参与训练源采样，不可改动。
    Build the training pool {dataset: data}; flipped copies are appended under keys such as "name@UD". The insertion order enters the source sampling and must not change."""
    pool_names = C.FOLDS + C.HOLDOUT
    need = set(pool_names) if not cond else (set(targets) | {f for f in pool_names if C.cond_of(f) == cond})
    folds = {f: ds.load_fold(f, goodch, data_dir, channels=channels, ablation=ablation) for f in pool_names if f in need}
    if with_flips:
        for f in [x for x in pool_names if x in need]:
            for fl in C.FLIP_TYPES:
                folds[f + "@" + fl] = ds.load_fold(f, goodch, data_dir, flip=fl, channels=channels, ablation=ablation)
    return folds


def fold_loss(model, folds, sup, sname, rng, huber, dev, n_channel, aug):
    """对一个源折采样 query 并计算损失。aug = (ramp_std, gain_std, offset_std)。 / Sample queries from one source dataset and compute the loss; aug = (ramp_std, gain_std, offset_std)."""
    sc = sup[sname]
    fd = folds[sname]
    rawccn = 2 * n_channel
    nch = 4 * n_channel
    qn = min(C.QBATCH, fd["N"] - fd["t0"])
    qidx = rng.randint(fd["t0"], fd["N"], size=qn)
    qxw = torch.from_numpy(ds.to_frames(fd["x"], qidx)).to(dev)
    ramp = torch.from_numpy((rng.standard_normal((qn, 1, rawccn)).astype(np.float32) * aug[0])).to(dev)
    qxw[:, :, :rawccn] = qxw[:, :, :rawccn] + ramp
    gg = torch.from_numpy((1 + rng.standard_normal((1, 1, n_channel)).astype(np.float32) * aug[1])).to(dev).repeat(1, 1, 2)
    g_aug = torch.ones(1, 1, nch, device=dev)
    g_aug[:, :, :rawccn] = gg
    oo = torch.from_numpy((rng.standard_normal((1, 1, n_channel)).astype(np.float32) * aug[2])).to(dev).repeat(1, 1, 2)
    o_aug = torch.zeros(1, 1, nch, device=dev)
    o_aug[:, :, :rawccn] = oo
    sxw_t = sc["xw"] * g_aug + o_aug
    qxw_t = qxw * g_aug + o_aug
    sy = sc["y"]
    qy = torch.from_numpy(fd["y"][qidx].astype(np.float32)).to(dev)
    zsf, zsh = model.encode(sxw_t)
    zqf, zqh = model.encode(qxw_t)
    pred = soft_readout(zqf[:, 0], zsf[:, 0], sy, model.logtau)
    loss_fit = ((pred - qy) ** 2).mean()
    aux = huber(soft_readout(zqh[0], zsh[0], sy, model.logtau), qy)
    return loss_fit + C.AUXW * aux


def train_one_step(model, opt, folds, sup, srcs, rng, huber, dev, n_channel, aug):
    loss = fold_loss(model, folds, sup, srcs[rng.randint(len(srcs))], rng, huber, dev, n_channel, aug)
    if not torch.isfinite(loss):
        opt.zero_grad()
        return
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), C.CLIP_NORM)
    opt.step()


@torch.no_grad()
def evaluate(model, fd, qlo, qhi, dev):
    """锚 = fd 当前 [s0,s1] 段的锚帧；对 query 帧 [qlo,qhi) 逐帧预测。返回 (preds, qidx)。 / Anchors from the current [s0,s1] segment of fd; predict the query frames [qlo,qhi) one by one. Returns (preds, qidx)."""
    sidx = ds.support_indices(fd)
    sxw = torch.from_numpy(ds.to_frames(fd["x"], sidx)).to(dev)
    sy = torch.from_numpy(fd["y"][sidx].astype(np.float32)).to(dev)
    zsf, _ = model.encode(sxw)
    qidx = np.arange(qlo, qhi)
    preds = np.empty(len(qidx), np.float32)
    for i in range(0, len(qidx), C.EVAL_BATCH):
        sub = qidx[i:i + C.EVAL_BATCH]
        qxw = torch.from_numpy(ds.to_frames(fd["x"], sub)).to(dev)
        zqf, _ = model.encode(qxw)
        preds[i:i + C.EVAL_BATCH] = soft_readout(zqf[:, 0], zsf[:, 0], sy, model.logtau).cpu().numpy()
    return preds, qidx


def train_fold(target, folds, huber, log, args, dev, n_channel):
    """训练并评测一个被测折。返回结果字典，并保存预测 .npy 与权重 .pt。 / Train and evaluate one dataset; returns the result dict and saves the .npy predictions and the .pt weights."""
    t_start = time.time()
    rng = np.random.RandomState(args.seed + (C.FOLDS.index(target) if target in C.FOLDS else 0))
    srcs = [f for f in folds if f != target and not f.startswith(target + "@")
            and (not args.cond or C.cond_of(f) == args.cond)]
    if not srcs:
        raise RuntimeError("%s: no training source (check --cond)" % target)
    sup = {}
    for sname in srcs:
        fd = folds[sname]
        sidx = ds.support_indices(fd)
        sup[sname] = dict(xw=torch.from_numpy(ds.to_frames(fd["x"], sidx)).to(dev),
                          y=torch.from_numpy(fd["y"][sidx].astype(np.float32)).to(dev))
    model = HingeNet(n_channel, drift_correction=(args.ablation != "nodrift")).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WD)
    aug = (0.0, 0.0, 0.0) if args.ablation in ("aug0", "noaug") else (C.RAMP_STD, C.GAIN_STD, C.OFFSET_STD)
    log("  [%s] cond=%s sources=%d" % (target, C.COND_NAME.get(args.cond, args.cond) if args.cond else "mixed", len(srcs)))
    fd_t = folds[target]
    best_mae, best_state, best_ep, pat = float("inf"), None, -1, 0
    for ep in range(args.epochs):
        model.train()
        for _ in range(C.EPOCH_STEPS):
            train_one_step(model, opt, folds, sup, srcs, rng, huber, dev, n_channel, aug)
        model.eval()
        preds_v, qv = evaluate(model, fd_t, fd_t["v0"], fd_t["t0"], dev)
        val_mae = float(np.mean(np.abs(preds_v - fd_t["y"][qv])))
        if val_mae < best_mae - 1e-6:
            best_mae, best_ep, pat = val_mae, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            preds_t, qt = evaluate(model, ds.fresh_anchor(fd_t), fd_t["t0"], fd_t["N"], dev)
            log("  [%s] NEWBEST ep%d val=%.4f test=%.4f" % (target, ep, val_mae, float(np.mean(np.abs(preds_t - fd_t["y"][qt])))))
        else:
            pat += 1
        log("  [%s] ep %d val %.4f best %.4f@ep%d pat %d (%.0fs)" % (target, ep, val_mae, best_mae, best_ep, pat, time.time() - t_start))
        if pat >= args.patience:
            log("  [%s] EARLYSTOP ep%d best %.4f@ep%d" % (target, ep, best_mae, best_ep))
            break
    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    model.eval()
    fd_t["s0"], fd_t["s1"] = fd_t["ts0"], fd_t["ts1"]          # 最终评测锚 = 鲜锚段 / final-test anchors = fresh segment
    preds, qidx = evaluate(model, fd_t, fd_t["t0"], fd_t["N"], dev)
    mae = float(np.mean(np.abs(preds - fd_t["y"][qidx])))
    np.save(os.path.join(args.out, target + ".npy"), preds)
    torch.save(best_state if best_state is not None else {k: v.detach().cpu() for k, v in model.state_dict().items()},
               os.path.join(args.out, target + "_model.pt"))
    log("*** %s test MAE %.3f (%.0fs)" % (target, mae, time.time() - t_start))
    return dict(fold=target, dataset_no=C.DATASET_NO.get(target), mae_test=round(mae, 3), mae_test_exact=mae,
                best_epoch=int(best_ep), epochs_run=int(ep + 1), val_mae_best=float(best_mae), n_sources=len(srcs),
                seed=int(args.seed), ablation=args.ablation_name or "none", n_channel=int(n_channel), mask_k=int(args.mask_k),
                train_time_s=round(time.time() - t_start, 1), device=str(dev))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="leave-one-dataset-out training / evaluation of the decoder (one fold per process)")
    ap.add_argument("--data-dir", default=C.DATA_DIR, help="directory of the parquet files")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--target", required=True, help="evaluated dataset, e.g. Data_01")
    ap.add_argument("--cond", default="", choices=[""] + list(C.COND_CLI), help="mounting condition of the training sources (noload / assembly); default: the condition of the evaluated dataset")
    ap.add_argument("--ablation", default="", choices=[""] + list(C.ABLATIONS), help="ablation configuration name (see config.ABLATIONS)")
    ap.add_argument("--channels", default="", help="channel subset (0-119, comma-separated); disables the global channel mask")
    ap.add_argument("--mask-k", type=int, default=None, help="channels kept by the global mask; default config.MASK_K (all channels for no_channel_mask)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    return ap.parse_args(argv)


def main(argv=None):
    from .runner import machine_info, format_machine_info
    args = parse_args(argv)
    set_determinism(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = args.target
    if target not in C.FOLDS:
        raise SystemExit("unknown dataset %r; evaluated datasets: %s" % (target, ", ".join(C.ORDER)))
    args.cond = C.COND_CLI[args.cond] if args.cond else C.cond_of(target)
    # 消融配置名 → 内部开关与掩膜数（命令行与产物只出现配置名）/ ablation name → internal switch and mask size (only the name appears outside)
    args.ablation_name = args.ablation
    switch, mask_default = C.ABLATIONS[args.ablation] if args.ablation else ("", C.MASK_K)
    args.ablation = switch
    if args.mask_k is None:
        args.mask_k = mask_default
    log("command: " + "python -m hingesense.trainer " + " ".join(sys.argv[1:]))
    for line in format_machine_info(machine_info()):
        log(line)
    channels = [int(s) for s in args.channels.split(",") if s.strip()] or None
    n_channel = len(channels) if channels else C.N_CHANNEL
    goodch = None if channels else ds.compute_channel_mask(C.FOLDS + C.HOLDOUT, args.data_dir, args.mask_k)
    if goodch is not None:
        log("channel mask top%d = %s" % (args.mask_k, goodch.tolist()))
    else:
        log("channel subset (%d channels) = %s" % (n_channel, channels))
    with_flips = args.ablation not in ("flip0", "noaug")
    log("=== %s seed=%d epochs=%d patience=%d ablation=%s channels=%d dev=%s ===" % (
        target, args.seed, args.epochs, args.patience, args.ablation_name or "none", n_channel, dev))
    folds = build_pool([target], goodch, args.cond, args.data_dir, channels, args.ablation, with_flips)
    log("pool=%d folds (flips=%s)" % (len(folds), with_flips))
    huber = nn.SmoothL1Loss(beta=C.HUBER_BETA)
    row = train_fold(target, folds, huber, log, args, dev, n_channel)
    with open(os.path.join(args.out, target + ".json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=1)
    log("==== %s DONE" % target)


if __name__ == "__main__":
    main()
