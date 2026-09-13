#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Electrode permutation importance: for every dataset of the decoder, the 15 channels of each electrode are shuffled along time in the test segment (10 times); the MAE increase is the importance.
When all 12 datasets are available the ranking file results/Fig_S8_4/electrode_ranking.txt is written (the only source of the reduced configuration and of the failure electrodes).
    python electrode_importance.py [--folds 1-12] [--jobs 12] [--gpus 0,1]
Requires results/raw/decoder/<dataset>_model.pt (python train.py).

电极置换重要性：对本文解码器的每个数据集，把每个电极的 15 个通道在测试段沿时间随机打乱（10 次），MAE 增量为重要性。
12 个数据集齐全后写出排名文件 results/Fig_S8_4/electrode_ranking.txt（电极精简配置与失效电极的唯一来源）。
需要先有 results/raw/decoder/<数据集>_model.pt（python train.py）。"""
import argparse
import datetime
import json
import os
import sys

from hingesense import config as C
from hingesense import check
from hingesense import ranking as R
from hingesense import runner


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--folds", default="all", help="dataset numbers 1-12 (e.g. 1,3,7-9) or names such as Data_03; default all")
    ap.add_argument("--model-dir", default=os.path.join(C.RAW_DIR, "decoder"), help="directory of the decoder weights")
    ap.add_argument("--out", default=os.path.join(C.RAW_DIR, "importance"), help="output directory")
    ap.add_argument("--jobs", type=int, default=1, help="processes run in parallel (default 1)")
    ap.add_argument("--gpus", default="", help="GPU ids to rotate over, e.g. 0,1; default all visible GPUs; 'none' for CPU only")
    ap.add_argument("--rerun", action="store_true", help="recompute datasets whose json already exists")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = ap.parse_args()
    started = datetime.datetime.now()
    folds = C.parse_folds(args.folds)
    check.verify_checksums()
    missing = [f for f in folds if not os.path.exists(os.path.join(args.model_dir, f + "_model.pt"))]
    if missing:
        sys.exit("decoder weights missing for %d dataset(s) (%s); run:  python train.py --folds %s" % (
            len(missing), ", ".join(str(C.DATASET_NO[f]) for f in missing), ",".join(str(C.DATASET_NO[f]) for f in missing)))
    out_rel = os.path.relpath(args.out, C.ROOT_DIR)
    model_rel = os.path.relpath(args.model_dir, C.ROOT_DIR)
    jobs = [dict(name="importance/" + f, gpu=True, log=os.path.join(args.out, f + ".log"),
                 cmd=runner.worker_cmd("evaluate", "importance", "--fold", f, "--model-dir", model_rel, "--json", os.path.join(out_rel, f + ".json")))
            for f in folds if args.rerun or not os.path.exists(os.path.join(args.out, f + ".json"))]
    print("%d job(s) to run; jobs=%d gpus=%s" % (len(jobs), args.jobs, runner.parse_gpus(args.gpus) or "none"))
    n_ok, failures = runner.run_jobs(jobs, args.jobs, runner.parse_gpus(args.gpus), dry_run=args.dry_run)
    if args.dry_run:
        return
    if failures:
        sys.exit("%d job(s) failed; see the log paths above" % len(failures))
    norm = {}
    for f in C.ORDER:
        p = os.path.join(args.out, f + ".json")
        if os.path.exists(p):
            d = json.load(open(p))["importance_norm"]
            norm[f] = [float(d["e%d" % e]) for e in range(C.N_ELECTRODE)]
    if len(norm) < len(C.ORDER):
        print("importance available for %d/12 datasets; the ranking file needs all 12 (missing: %s)" % (
            len(norm), ",".join(str(C.DATASET_NO[f]) for f in C.ORDER if f not in norm)))
        return
    rows = R.build_ranking(norm)
    R.write_ranking(C.RANKING_FILE, rows, "python " + " ".join(os.path.relpath(a, C.ROOT_DIR) if os.path.isabs(a) else a for a in sys.argv))
    print("electrode ranking → %s" % os.path.relpath(C.RANKING_FILE, C.ROOT_DIR))
    print("top-%d electrodes: %s;  failure electrodes (top-%d): %s" % (
        C.SUBSET_ELECTRODES, ", ".join("E%d" % e for e in R.top_electrodes(C.SUBSET_ELECTRODES)), C.FAILURE_ELECTRODES,
        ", ".join("E%d" % e for e in R.top_electrodes(C.FAILURE_ELECTRODES))))
    print("elapsed %.0f s" % (datetime.datetime.now() - started).total_seconds())


if __name__ == "__main__":
    main()
