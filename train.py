#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate: the proposed decoder (default) or a comparison method, leave-one-dataset-out.

    python train.py                                  # proposed decoder, 12 datasets
    python train.py --folds 1,3,7-9                  # only these datasets (paper numbering)
    python train.py --method gru                     # one comparison method
    python train.py --method all --jobs 24 --gpus 0,1   # the 13 comparison methods of the paper + the decoder
    python train.py --method gru_aug                 # augmented variants (gru_aug / tcn_aug / rf_aug) run by name only
    python train.py --ablation no_drift_correction   # one ablation (names: see --help)
    python train.py --electrodes top4                # reduced electrode configuration (top-4 of the ranking file)
    python train.py --seed 2028                      # another random seed
Every (method × dataset) is one process writing results/raw/<configuration>/<dataset>.{json,npy,_model.pt,log}; datasets whose json exists are skipped.

训练与评估：本文方法（默认）或对比方法，按数据集留一评测。用法同上：默认本文方法 12 个数据集；--folds 选数据集（文章编号）；
--method 选一种对比方法，all 为文章的 13 种对比方法 + 本文方法，增强变体（gru_aug / tcn_aug / rf_aug）只能按名单独运行；
--ablation 一个消融配置；--electrodes top4 电极精简配置（排名文件的前 4 个电极）；--seed 换随机种子。
每个（方法 × 数据集）是一个独立进程，产物写入 results/raw/<配置>/<折名>.{json,npy,_model.pt,log}；已有 json 的折默认跳过。
"""
import argparse
import datetime
import json
import os
import sys

from hingesense import config as C
from hingesense import check
from hingesense import ranking as R
from hingesense import runner

METHODS = C.BASELINE_METHODS
CPU_METHODS = C.CPU_METHODS
METHOD_ORDER = list(C.PAPER_METHODS)          # --method all 的范围与运行顺序（文章 Table II 的 13 种）/ scope and order of --method all (the 13 methods of Table II)


def decoder_dir_name(args, electrodes):
    if args.ablation:
        name = "ablation_" + args.ablation
    elif electrodes is not None:
        name = "electrodes_" + (args.electrodes.lower() if args.electrodes.lower().startswith("top") else C.electrodes_tag(electrodes))
    else:
        name = "decoder"
    if args.seed != C.SEED:
        name += "_seed%d" % args.seed
    return name


def decoder_jobs(args, folds, electrodes, out):
    jobs = []
    for f in folds:
        if os.path.exists(os.path.join(out, f + ".json")) and not args.rerun:
            continue
        extra = ["--seed", args.seed, "--epochs", args.epochs, "--patience", args.patience]
        if args.ablation:
            extra += ["--ablation", args.ablation]
        if electrodes is not None:
            extra += ["--channels", ",".join(str(c) for c in C.electrodes_to_channels(electrodes))]
        jobs.append(dict(name="%s/%s" % (os.path.basename(out), f), gpu=True, log=os.path.join(out, f + ".log"),
                         cmd=runner.worker_cmd("trainer", "--out", os.path.relpath(out, C.ROOT_DIR), "--target", f, *extra)))
    return jobs


def baseline_jobs(args, folds, method):
    out = os.path.join(C.RAW_DIR, "baselines", method)
    jobs = []
    for f in folds:
        if os.path.exists(os.path.join(out, f + ".json")) and not args.rerun:
            continue
        jobs.append(dict(name="baselines/%s/%s" % (method, f), gpu=method not in CPU_METHODS, log=os.path.join(out, f + ".log"),
                         cmd=runner.worker_cmd("baselines", "--method", method, "--fold", f, "--out", os.path.relpath(out, C.ROOT_DIR), "--epochs", args.epochs, "--patience", args.patience)))
    return jobs


def report_decoder(out, folds):
    """一行汇总：已有结果的数据集数、均值、最差、亚度个数 / one summary line: datasets done, mean, worst, sub-degree count."""
    vals = [json.load(open(os.path.join(out, f + ".json")))["mae_test_exact"] for f in folds if os.path.exists(os.path.join(out, f + ".json"))]
    if vals:
        print("  %d/%d datasets  mean %.3f°  worst %.3f°  sub-degree %d/%d" % (len(vals), len(folds), sum(vals) / len(vals), max(vals), sum(v < 1 for v in vals), len(vals)))


def report_baseline(method, folds):
    """一行汇总：顺序 / 块乱序 MAE 的均值 / one summary line: mean sequential / block-shuffled MAE."""
    out = os.path.join(C.RAW_DIR, "baselines", method)
    rows = [json.load(open(os.path.join(out, f + ".json"))) for f in folds if os.path.exists(os.path.join(out, f + ".json"))]
    if rows:
        print("  %d/%d datasets  mean MAE %.2f° (sequential)  %.2f° (block-shuffled)" % (
            len(rows), len(folds), sum(r["mae_seq_exact"] for r in rows) / len(rows), sum(r["mae_shuf_exact"] for r in rows) / len(rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--method", default="ours", help="ours (default); all = the 13 comparison methods of the paper plus ours; or one method: %s" % ", ".join(METHOD_ORDER + list(C.AUGMENTED_METHODS)))
    ap.add_argument("--folds", default="all", help="dataset numbers 1-12 (e.g. 1,3,7-9) or names such as Data_03; default all")
    ap.add_argument("--ablation", default="", choices=[""] + list(C.ABLATIONS), help="ablation configuration (ours only)")
    ap.add_argument("--electrodes", default="", help="electrode subset: topK from %s or a list such as 0,1,8,15 (ours only)" % os.path.relpath(C.RANKING_FILE, C.ROOT_DIR))
    ap.add_argument("--seed", type=int, default=C.SEED, help="global seed (ours only); default %d" % C.SEED)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--jobs", type=int, default=1, help="processes run in parallel (default 1)")
    ap.add_argument("--gpus", default="", help="GPU ids to rotate over, e.g. 0,1; default all visible GPUs; 'none' for CPU only")
    ap.add_argument("--rerun", action="store_true", help="recompute folds whose json already exists")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = ap.parse_args()
    started = datetime.datetime.now()
    if args.method != "ours" and (args.ablation or args.electrodes or args.seed != C.SEED):
        sys.exit("--ablation / --electrodes / --seed apply to --method ours only")
    if args.ablation and args.electrodes:
        sys.exit("choose either --ablation or --electrodes")
    folds = C.parse_folds(args.folds)
    check.verify_checksums()
    electrodes = R.parse_electrodes(args.electrodes) if args.electrodes else None
    jobs, plan = [], []
    if args.method in ("ours", "all"):
        out = os.path.join(C.RAW_DIR, decoder_dir_name(args, electrodes))
        new_jobs = decoder_jobs(args, folds, electrodes, out)
        if new_jobs and not args.dry_run:      # 只有真正训练时才写配置文件，跳过已有结果的调用不改写它 / written only when something is trained
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "configuration.json"), "w", encoding="utf-8") as f:
                json.dump(dict(method="ours", ablation=args.ablation or None, electrodes=electrodes,
                               channels=(C.electrodes_to_channels(electrodes) if electrodes is not None else None), seed=args.seed,
                               epochs=args.epochs, patience=args.patience,
                               command="python " + " ".join(os.path.relpath(a, C.ROOT_DIR) if os.path.isabs(a) else a for a in sys.argv)), f, indent=1)
        jobs += new_jobs
        plan.append(("ours", out))
    if args.method == "all":
        methods = METHOD_ORDER
    elif args.method == "ours":
        methods = []
    elif args.method in METHODS:
        methods = [args.method]
    else:
        sys.exit("unknown method %r; choose ours, all or one of %s" % (args.method, ", ".join(METHODS)))
    for m in methods:
        jobs += baseline_jobs(args, folds, m)
        plan.append((m, os.path.join(C.RAW_DIR, "baselines", m)))
    print("%d job(s) to run (%d already done and skipped); jobs=%d gpus=%s" % (
        len(jobs), sum(len(folds) for _ in plan) - len(jobs), args.jobs, runner.parse_gpus(args.gpus) or "none"))
    n_ok, failures = runner.run_jobs(jobs, args.jobs, runner.parse_gpus(args.gpus), dry_run=args.dry_run)
    if args.dry_run:
        return
    for m, out in plan:
        print("%s → %s" % (m, os.path.relpath(out, C.ROOT_DIR)))
        if m == "ours":
            report_decoder(out, folds)
        else:
            report_baseline(m, folds)
    print("elapsed %.0f s" % (datetime.datetime.now() - started).total_seconds())
    if failures:
        sys.exit("%d job(s) failed; see the log paths above" % len(failures))


if __name__ == "__main__":
    main()
