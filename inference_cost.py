#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model size and batch-1 latency of the decoder (Table S2.3 and the numbers of Section S2.3), measured on this machine.
    python inference_cost.py                        # one CPU thread, plus the GPU when available
    python inference_cost.py --device cpu --threads 1
    python inference_cost.py --device cuda:0
Outputs: results/raw/inference_cost/decoder_cpu<threads>.json and decoder_gpu.json (platform, PyTorch version, latency statistics, parameter count, ...).
Requires results/raw/decoder/<dataset>_model.pt (python train.py, at least --folds 1).

本文解码器的模型体量与批 1 时延（Table S2.3 与 S2.3 正文的数字），在本机上测量：默认测单个 CPU 线程，有 GPU 时再测 GPU。
产物：results/raw/inference_cost/decoder_cpu<线程数>.json、decoder_gpu.json（含平台、PyTorch 版本、时延统计、参数量等）。
需要先有 results/raw/decoder/<数据集>_model.pt（python train.py，至少 --folds 1）。"""
import argparse
import datetime
import json
import os
import sys

from hingesense import config as C
from hingesense import check
from hingesense import runner


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--device", default="auto", help="cpu, cuda:0, or auto (default: cpu with one thread, plus cuda:0 when available)")
    ap.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads when measuring on the CPU (default 1)")
    ap.add_argument("--iters", type=int, default=0, help="timed queries after 200 warm-up queries (default 1000 on CPU, 2000 on GPU)")
    ap.add_argument("--fold", default="1", help="dataset whose weights and test frames are used (default 1)")
    ap.add_argument("--model-dir", default=os.path.join(C.RAW_DIR, "decoder"), help="directory of the decoder weights")
    ap.add_argument("--out", default=os.path.join(C.RAW_DIR, "inference_cost"), help="output directory")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = ap.parse_args()
    started = datetime.datetime.now()
    check.verify_checksums()
    fold = C.parse_folds(args.fold)[0]
    if not os.path.exists(os.path.join(args.model_dir, fold + "_model.pt")):
        sys.exit("decoder weights missing for dataset %d; run:  python train.py --folds %d" % (C.DATASET_NO[fold], C.DATASET_NO[fold]))
    if args.device == "auto":
        devices = ["cpu"] + (["cuda:0"] if runner.parse_gpus("") else [])
    else:
        devices = [args.device]
    os.makedirs(args.out, exist_ok=True)
    jobs, outs = [], []
    for dev in devices:
        gpu = dev.startswith("cuda")
        tag = "gpu" if gpu else "cpu%d" % args.threads
        out = os.path.join(args.out, "decoder_%s.json" % tag)
        iters = args.iters or (2000 if gpu else 1000)
        jobs.append(dict(name="inference_cost/decoder_" + tag, gpu=gpu, log=out[:-5] + ".log",
                         cmd=runner.worker_cmd("cost", "--model-dir", os.path.relpath(args.model_dir, C.ROOT_DIR), "--fold", fold, "--device", dev,
                                               "--threads", args.threads if not gpu else 0, "--iters", iters, "--out", os.path.relpath(out, C.ROOT_DIR))))
        outs.append(out)
    n_ok, failures = runner.run_jobs(jobs, 1, runner.parse_gpus(""), dry_run=args.dry_run)
    if args.dry_run:
        return
    if failures:
        sys.exit("%d job(s) failed; see the log paths above" % len(failures))
    for out in outs:
        d = json.load(open(out))
        lat = d["latency_batch1_ms"]
        print("%s (PyTorch %s): batch-1 latency median %.3f ms, P95 %.3f ms (%d timed queries, %d anchors); params %d, %.3f MiB, %d encoding MACs" % (
            d["info"]["platform"], d["info"]["pytorch"], lat["median_ms"], lat["p95_ms"], lat["n"], d["compute"]["n_anchor"],
            d["model"]["params"], d["model"]["size_fp32_MiB"], d["compute"]["mac_encode"]))
    print("results → %s   (table: python make_tables.py --only Table_S2_3)" % os.path.relpath(args.out, C.ROOT_DIR))
    print("elapsed %.0f s" % (datetime.datetime.now() - started).total_seconds())


if __name__ == "__main__":
    main()
