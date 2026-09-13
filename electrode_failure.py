#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-electrode failure test: all channels of one electrode are set to zero at test time, without retraining. By default both the 16-electrode
model and the reduced top-4 model are tested with each of the three highest-ranked electrodes (from results/Fig_S8_4/electrode_ranking.txt).
    python electrode_failure.py [--config all|full|top4] [--electrodes top3|0,8,15] [--folds 1-12] [--jobs 8] [--gpus 0,1]
Requires python train.py (16-electrode model), python electrode_importance.py (ranking file) and python train.py --electrodes top4 (reduced model).

单电极失效测试：测试时把一个电极的全部通道置零，模型不重训。默认对 16 电极模型与排名前 4 电极的精简模型，
各测试排名前 3 个电极（电极来自 results/Fig_S8_4/electrode_ranking.txt）。
需要先有 python train.py（16 电极模型）、python electrode_importance.py（排名文件）与 python train.py --electrodes top4（精简模型）。"""
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
    ap.add_argument("--config", default="all", choices=["all", "full", "top4"], help="which trained models to test: full = 16 electrodes, top4 = the reduced configuration (default both)")
    ap.add_argument("--electrodes", default="top%d" % C.FAILURE_ELECTRODES, help="electrodes to disable: topK from the ranking file or a list such as 0,8,15")
    ap.add_argument("--folds", default="all", help="dataset numbers 1-12 (e.g. 1,3,7-9) or names such as Data_03; default all")
    ap.add_argument("--out", default=os.path.join(C.RAW_DIR, "electrode_failure"), help="output directory")
    ap.add_argument("--jobs", type=int, default=1, help="processes run in parallel (default 1)")
    ap.add_argument("--gpus", default="", help="GPU ids to rotate over, e.g. 0,1; default all visible GPUs; 'none' for CPU only")
    ap.add_argument("--rerun", action="store_true", help="recompute tests whose json already exists")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = ap.parse_args()
    started = datetime.datetime.now()
    folds = C.parse_folds(args.folds)
    check.verify_checksums()
    electrodes = R.parse_electrodes(args.electrodes)
    configs = []
    if args.config in ("all", "full"):
        configs.append(("decoder", os.path.join(C.RAW_DIR, "decoder"), None))
    if args.config in ("all", "top4"):
        mdir = os.path.join(C.RAW_DIR, "electrodes_top4")
        cfg = os.path.join(mdir, "configuration.json")
        channels = json.load(open(cfg))["channels"] if os.path.exists(cfg) else C.electrodes_to_channels(R.top_electrodes(C.SUBSET_ELECTRODES))
        configs.append(("electrodes_top4", mdir, channels))
    jobs = []
    for name, mdir, channels in configs:
        missing = [f for f in folds if not os.path.exists(os.path.join(mdir, f + "_model.pt"))]
        if missing:
            hint = "python train.py" if name == "decoder" else "python train.py --electrodes top4"
            sys.exit("%s: weights missing for dataset(s) %s; run:  %s --folds %s" % (
                name, ",".join(str(C.DATASET_NO[f]) for f in missing), hint, ",".join(str(C.DATASET_NO[f]) for f in missing)))
        for f in folds:
            for e in electrodes:
                jpath = os.path.join(args.out, name, "%s_E%d.json" % (f, e))
                if os.path.exists(jpath) and not args.rerun:
                    continue
                extra = ["--channels", ",".join(str(c) for c in channels)] if channels else []
                jobs.append(dict(name="electrode_failure/%s/%s/E%d" % (name, f, e), gpu=True, log=jpath[:-5] + ".log",
                                 cmd=runner.worker_cmd("evaluate", "electrode_failure", "--fold", f, "--model-dir", os.path.relpath(mdir, C.ROOT_DIR),
                                                       "--electrode", e, "--json", os.path.relpath(jpath, C.ROOT_DIR), *extra)))
    print("%d job(s) to run for electrodes %s; jobs=%d gpus=%s" % (len(jobs), ", ".join("E%d" % e for e in electrodes), args.jobs, runner.parse_gpus(args.gpus) or "none"))
    n_ok, failures = runner.run_jobs(jobs, args.jobs, runner.parse_gpus(args.gpus), dry_run=args.dry_run)
    if args.dry_run:
        return
    for name, _, _ in configs:
        for e in electrodes:
            vals = []
            for f in folds:
                p = os.path.join(args.out, name, "%s_E%d.json" % (f, e))
                if os.path.exists(p):
                    vals.append(json.load(open(p))["mae_failure_exact"])
            if vals:
                print("  %-16s E%-2d failure: mean MAE %.3f° over %d dataset(s)" % (name, e, sum(vals) / len(vals), len(vals)))
    print("elapsed %.0f s" % (datetime.datetime.now() - started).total_seconds())
    if failures:
        sys.exit("%d job(s) failed; see the log paths above" % len(failures))


if __name__ == "__main__":
    main()
