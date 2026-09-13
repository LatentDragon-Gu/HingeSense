# -*- coding: utf-8 -*-
"""基于训练产物（模型权重）的后评估：单电极失效、置换重要性。

    electrode_failure  测试期把某电极的全部通道置零（模型不重训），16 电极与精简电极配置的模型均可
    importance         置换检验：把某电极 15 个通道在测试段沿时间随机打乱（10 个种子），MAE 增量为重要性
单折工作进程，由根目录 electrode_failure.py / electrode_importance.py 调度，也可直接运行：
    python -m hingesense.evaluate electrode_failure --fold Data_01 --model-dir results/raw/decoder --electrode 0 --json out.json
    python -m hingesense.evaluate electrode_failure --fold Data_01 --model-dir results/raw/electrodes_top4 --channels <通道列表> --electrode 8 --json out.json
    python -m hingesense.evaluate importance        --fold Data_01 --model-dir results/raw/decoder --json out.json
模型文件为 <model-dir>/<fold>_model.pt（trainer 的产物）。

Post-hoc evaluation of trained weights: electrode failure and permutation importance.
    electrode_failure  all channels of one electrode set to zero at test time (no retraining), for the 16-electrode and the reduced model
    importance         permutation test: the 15 channels of one electrode shuffled along time in the test segment (10 seeds), MAE increase = importance
One dataset per process, scheduled by electrode_failure.py / electrode_importance.py or run directly as shown above.
The model file is <model-dir>/<fold>_model.pt written by trainer.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from . import config as C
from . import dataset as ds
from .model import HingeNet
from .trainer import evaluate as predict_segment, set_determinism


def load_model(model_dir, fold, n_channel, dev):
    path = os.path.join(model_dir, fold + "_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError("%s not found: train the decoder for this dataset first (python train.py --folds ...)" % path)
    model = HingeNet(n_channel).to(dev)
    model.load_state_dict(torch.load(path, map_location=dev))
    return model.eval()


def test_mae(model, fd, dev):
    """鲜锚 + 测试段逐帧预测 → (MAE, preds, qidx)。 / Fresh anchors + frame-wise test predictions → (MAE, preds, qidx)."""
    preds, qidx = predict_segment(model, ds.fresh_anchor(fd), fd["t0"], fd["N"], dev)
    return float(np.mean(np.abs(preds - fd["y"][qidx]))), preds, qidx


# ====================================================================== 单电极失效 / electrode failure
def zero_electrode(fd, electrode, channels=None):
    """把含该电极的通道在四条特征通路中全部置零。channels 给定时为子集模型的列映射。 / Zero the channels of this electrode in all four feature pathways; channels gives the column mapping of a reduced model."""
    ch = set(ds.electrode_channels(electrode))
    if channels is None:
        n = C.N_CHANNEL
        cols = [c for c in range(C.N_CHANNEL) if c in ch]
    else:
        n = len(channels)
        cols = [p for p, c in enumerate(channels) if c in ch]
    x = fd["x"].copy()
    for blk in range(4):
        for p in cols:
            x[:, blk * n + p] = 0.0
    out = dict(fd)
    out["x"] = x
    return out, len(cols), n


def failure_fold(fold, model_dir, electrode, goodch, data_dir, channels, dev):
    fd = ds.load_fold(fold, goodch, data_dir, channels=channels)
    model = load_model(model_dir, fold, len(channels) if channels else C.N_CHANNEL, dev)
    mae0, _, _ = test_mae(model, fd, dev)
    fdk, nk, ntot = zero_electrode(fd, electrode, channels)
    maek, _, _ = test_mae(model, fdk, dev)
    return dict(configuration=("subset" if channels else "full"), fold=fold, dataset_no=C.DATASET_NO.get(fold), electrode=int(electrode),
                n_channels_zeroed=nk, n_channels_total=ntot, channels=(list(channels) if channels else None),
                mae_test=round(mae0, 4), mae_failure=round(maek, 4), delta=round(maek - mae0, 4),
                mae_test_exact=mae0, mae_failure_exact=maek)


# ====================================================================== 置换重要性 / permutation importance
def importance_fold(fold, model_dir, goodch, data_dir, dev, per_seed=False, log=print):
    z = ds.load_raw(fold, data_dir)
    x_raw0 = z["x_clean"].astype(np.float32)
    x_ccn0 = ds.get_ccn(fold, data_dir)
    fd0 = ds.load_fold(fold, goodch, data_dir)
    model = load_model(model_dir, fold, C.N_CHANNEL, dev)
    mae_base, _, _ = test_mae(model, fd0, dev)
    t0, N = fd0["t0"], fd0["N"]
    importance, per = {}, {}
    t_start = time.time()
    for e in range(C.N_ELECTRODE):
        chans = ds.electrode_channels(e)
        maes = []
        for s in range(C.IMPORTANCE_SEEDS):
            perm = np.random.RandomState(7000 + 100 * e + s).permutation(N - t0)
            x_raw = x_raw0.copy()
            x_ccn = x_ccn0.copy()
            for ch in chans:
                x_ccn[t0:N, ch] = x_ccn[t0:N, ch][perm]
                x_raw[t0:N, ch] = x_raw[t0:N, ch][perm]
            fd = ds.load_fold(fold, goodch, data_dir, x_raw=x_raw, x_ccn=x_ccn)
            m, _, _ = test_mae(model, fd, dev)
            maes.append(m)
        importance["e%d" % e] = round(float(np.mean(maes)) - mae_base, 4)
        per["e%d" % e] = [round(m, 4) for m in maes]
        log("  [%s] e%-2d mean_perm=%.4f imp=%+.4f (%.0fs)" % (fold, e, np.mean(maes), importance["e%d" % e], time.time() - t_start))
    vals = np.array([importance["e%d" % e] for e in range(C.N_ELECTRODE)])
    mx = float(vals.max())
    norm = {("e%d" % e): (round(importance["e%d" % e] / mx, 4) if mx > 0 else importance["e%d" % e]) for e in range(C.N_ELECTRODE)}
    row = {"fold": fold, "dataset_no": C.DATASET_NO.get(fold), "mae_test": round(mae_base, 4), "permutations": C.IMPORTANCE_SEEDS,
           "importance": importance, "importance_norm": norm}
    if per_seed:
        row["per_seed"] = per
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description="post-hoc evaluation of trained decoders (one dataset per process)")
    ap.add_argument("task", choices=["electrode_failure", "importance"])
    ap.add_argument("--fold", required=True, help="evaluated dataset, e.g. Data_01")
    ap.add_argument("--model-dir", required=True, help="directory holding <fold>_model.pt")
    ap.add_argument("--json", required=True, help="output json file")
    ap.add_argument("--data-dir", default=C.DATA_DIR)
    ap.add_argument("--mask-k", type=int, default=C.MASK_K)
    ap.add_argument("--channels", default="", help="channel list of a reduced-electrode model (electrode failure of the reduced configuration)")
    ap.add_argument("--electrode", type=int, default=0, help="electrode to disable (0-15)")
    ap.add_argument("--per-seed", action="store_true", help="also store the MAE of every permutation")
    args = ap.parse_args(argv)
    set_determinism(C.SEED)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    channels = [int(s) for s in args.channels.split(",") if s.strip()] or None
    goodch = None if channels else ds.compute_channel_mask(C.FOLDS + C.HOLDOUT, args.data_dir, args.mask_k)
    print("command: " + "python -m hingesense.evaluate " + " ".join(sys.argv[1:]), flush=True)
    if args.task == "electrode_failure":
        row = failure_fold(args.fold, args.model_dir, args.electrode, goodch, args.data_dir, channels, dev)
    else:
        row = importance_fold(args.fold, args.model_dir, goodch, args.data_dir, dev, args.per_seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=1, ensure_ascii=False)
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
