# -*- coding: utf-8 -*-
"""本文解码器的模型体量与批 1 时延（Table S2.3 与 S2.3 正文引用的数字）。由根目录 inference_cost.py 调用，也可直接运行：
    python -m hingesense.cost --model-dir results/raw/decoder --fold Data_01 --device cpu --threads 1 --iters 1000 --out results/raw/inference_cost/decoder_cpu1.json
    python -m hingesense.cost --model-dir results/raw/decoder --fold Data_01 --device cuda:0 --iters 2000 --out results/raw/inference_cost/decoder_gpu.json

测量口径：训练好的模型（默认数据集 1），鲜锚段的锚已预先编码并缓存，批 1；200 次热身后计时 iters 次查询；
每次查询从已构造好的 480 维特征向量开始，到返回标量角度结束（含张量准备、必要的设备传输、查询编码、锚匹配与角度插值），
不含采集、特征构造与锚缓存准备；模型处于评估模式、无梯度；CPU 用单个 PyTorch 线程，GPU 计时含同步。
另外给出参数量、FP32 体量、查询编码的线性层 MAC 数、锚字典体量与逐帧因果特征更新的 CPU 耗时。

Model size and batch-1 latency of the decoder (Table S2.3 and the numbers quoted in Section S2.3). Called by inference_cost.py.
Protocol: trained model (dataset 1 by default), anchors of the fresh segment encoded and cached in advance, batch size one;
200 warm-up queries, then iters timed queries; a query starts from a prepared 480-D feature vector and ends when the scalar
angle is returned (tensor preparation, device transfer where applicable, query encoding, anchor matching, angle interpolation),
excluding acquisition, feature construction and anchor caching; evaluation mode, no gradients; one PyTorch thread on the CPU,
synchronised timing on the GPU. Also reports the parameter count, FP32 size, linear-layer MACs of the query encoding, the size
of the anchor dictionary and the CPU time of the per-frame causal feature update.
"""
import argparse
import json
import os
import platform
import re
import sys
import statistics
import time

import numpy as np
import torch
from torch import nn

from . import config as C
from . import dataset as ds
from .model import HingeNet, soft_readout, count_parameters
from .trainer import set_determinism


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _stats(lat_ms):
    lat = sorted(lat_ms)
    return dict(n=len(lat), median_ms=statistics.median(lat), mean_ms=statistics.fmean(lat),
                p95_ms=lat[int(0.95 * (len(lat) - 1))], p99_ms=lat[int(0.99 * (len(lat) - 1))])


def _timeit(fn, iters, dev, warmup=200):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        _sync(dev)
        lat = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            _sync(dev)
            lat.append((time.perf_counter() - t0) * 1e3)
    return _stats(lat)


def _cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "CPU"


def _clean_cpu_name(name):
    """去掉商标符号、代际前缀与主频后缀，如 "11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz" → "Intel Core i7-11800H"。 / Strip trademark signs, generation prefix and clock suffix from the CPU name."""
    name = re.sub(r"\((R|TM|tm|r)\)", "", name)
    name = re.sub(r"^\d+(st|nd|rd|th) Gen ", "", name.strip())
    name = name.split("@")[0]
    return re.sub(r"\s+", " ", name).strip()


def platform_label(dev, threads):
    """Table S2.3 的 Platform 列：CPU 写型号与线程数，GPU 写型号。 / Platform column of Table S2.3: CPU model and thread count, or GPU model."""
    if dev.type == "cuda":
        g = torch.cuda.get_device_name(dev)
        return g if g.endswith("GPU") else g + " GPU"
    cpu = _clean_cpu_name(_cpu_model())
    return "%s CPU, %s" % (cpu, "one thread" if threads == 1 else "%d threads" % threads)


def _host_info(dev, threads):
    return dict(platform=platform_label(dev, threads), pytorch=torch.__version__, python=platform.python_version(), cuda=torch.version.cuda or "",
                device=str(dev), threads=threads, cpu=_cpu_model(), gpu=(torch.cuda.get_device_name(dev) if dev.type == "cuda" else ""))


def mac_count(model, call):
    """通过前向钩子统计线性层的乘加次数（不含逐元素运算与锚匹配）。 / Count linear-layer multiply-accumulates with forward hooks (element-wise operations and anchor matching excluded)."""
    tot = {"mac": 0}

    def hook(mod, inp, out):
        if isinstance(mod, nn.Linear):
            tot["mac"] += mod.in_features * mod.out_features * int(np.prod(out.shape[:-1]))

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        call()
    for h in hs:
        h.remove()
    return int(tot["mac"])


def feature_step_timer(fold, goodch, data_dir, n=2000):
    """逐帧在线特征递推（稳健白化 → 因果参考一步 → 参考白化 → 两条 EMA 差分）的 CPU 耗时。 / CPU time of the per-frame feature update (robust whitening → one causal reference step → reference whitening → two EMA differences)."""
    z = ds.load_raw(fold, data_dir)
    x_raw = z["x_clean"].astype(np.float64)
    s0, s1 = int(z["s0"]), int(z["s1"])
    sup = x_raw[s0:s1 + 1]
    med = np.median(sup, 0)
    iqr = np.percentile(sup, 75, 0) - np.percentile(sup, 25, 0) + 1e-6
    bad = np.ones(C.N_CHANNEL, bool)
    bad[goodch] = False
    c0 = np.nanmedian(sup, 0)
    iqr0 = ds._iqr_cols(sup)
    slen = s1 - s0 + 1
    half = max(slen // 2, 1)
    wob = np.abs(ds._iqr_cols(sup[:half]) - ds._iqr_cols(sup[half:])) / iqr0
    db = np.clip(3.0 * wob, 0.25, 1.0)
    lo_db, hi_db = 1.0 / (1.0 + db), 1.0 + db
    a_ccn = 1.0 - 2.0 ** (-1.0 / max(C.CCN_HALFLIFE_MULT * slen, 1.0))
    lo, hi = c0 - 5 * iqr0, c0 + 5 * iqr0
    csup = ds.get_ccn(fold, data_dir)[s0:s1 + 1]
    cmed = np.median(csup, 0)
    ciqr = np.percentile(csup, 75, 0) - np.percentile(csup, 25, 0) + 1e-6
    st = dict(c=c0.copy(), ad=iqr0 * 0.591, m16=None, m64=None)
    a16, a64 = 1.0 - 2.0 ** (-1.0 / 16), 1.0 - 2.0 ** (-1.0 / 64)

    def step(row):
        rawz = np.clip((row - med) / iqr, -C.CLIP_Z, C.CLIP_Z)
        rawz[bad] = 0.0
        cc = np.clip(st["c"], lo, hi)
        ratio = st["ad"] * 1.692 / iqr0
        eff = iqr0.copy()
        hiM, loM = ratio > hi_db, ratio < lo_db
        eff[hiM] = iqr0[hiM] * (ratio[hiM] / hi_db[hiM])
        eff[loM] = iqr0[loM] * (ratio[loM] / lo_db[loM])
        eff = np.clip(eff, 0.5 * iqr0, 2.0 * iqr0)
        ccn = (row - cc) / eff
        st["c"] = (1 - a_ccn) * st["c"] + a_ccn * row
        st["ad"] = (1 - a_ccn) * st["ad"] + a_ccn * np.abs(row - cc)
        ccnz = np.clip((ccn - cmed) / ciqr, -C.CLIP_Z, C.CLIP_Z)
        ccnz[bad] = 0.0
        if st["m16"] is None:
            st["m16"], st["m64"] = rawz.copy(), rawz.copy()
        st["m16"] = (1 - a16) * st["m16"] + a16 * rawz
        st["m64"] = (1 - a64) * st["m64"] + a64 * rawz
        return np.concatenate([rawz, ccnz, rawz - st["m16"], rawz - st["m64"]]).astype(np.float32)

    for i in range(200):
        step(x_raw[i])
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        step(x_raw[i % len(x_raw)])
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return dict(n=len(ts), median_us=statistics.median(ts), p95_us=ts[int(0.95 * (len(ts) - 1))], mean_us=statistics.fmean(ts))


def bench_decoder(args):
    dev = torch.device(args.device if (args.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    threads = torch.get_num_threads()
    goodch = ds.compute_channel_mask(C.FOLDS + C.HOLDOUT, args.data_dir, C.MASK_K)
    model = HingeNet().to(dev).eval()
    pt = os.path.join(args.model_dir, args.fold + "_model.pt")
    if not os.path.exists(pt):
        raise FileNotFoundError("%s not found: train the decoder for this dataset first (python train.py --folds %d)" % (pt, C.DATASET_NO[args.fold]))
    model.load_state_dict(torch.load(pt, map_location=dev))
    info = _host_info(dev, threads)
    info["weights"] = os.path.relpath(pt, C.ROOT_DIR)
    n_params = count_parameters(model)
    by_module = {name: sum(p.numel() for p in m.parameters()) for name, m in model.named_children()}
    by_module["logtau"] = 1
    fd = ds.load_fold(args.fold, goodch, args.data_dir)
    sidx = ds.support_indices(ds.fresh_anchor(fd))
    sxw = torch.from_numpy(ds.to_frames(fd["x"], sidx)).to(dev)
    sy = torch.from_numpy(fd["y"][sidx].astype(np.float32)).to(dev)
    with torch.no_grad():
        zs = model.encode(sxw)[0][:, 0].contiguous()
    test_idx = np.arange(fd["t0"], fd["N"])
    xs = fd["x"]
    q1 = torch.from_numpy(ds.to_frames(xs, test_idx[:1])).to(dev)
    mac_enc = mac_count(model, lambda: model.encode(q1))

    def end_to_end(i):
        q = torch.from_numpy(np.ascontiguousarray(xs[i:i + 1][:, None, :])).to(dev)
        return soft_readout(model.encode(q)[0][:, 0], zs, sy, model.logtau).item()

    idxs = np.resize(test_idx, args.iters)
    with torch.no_grad():
        for i in test_idx[:200]:
            end_to_end(int(i))
        _sync(dev)
        lat = []
        for i in idxs:
            t0 = time.perf_counter()
            end_to_end(int(i))
            _sync(dev)
            lat.append((time.perf_counter() - t0) * 1e3)
    e2e = _stats(lat)
    compute_only = _timeit(lambda: soft_readout(model.encode(q1)[0][:, 0], zs, sy, model.logtau), args.iters, dev)
    thr = {}
    with torch.no_grad():
        for B in (1, 32, 256, 1536):
            q = torch.from_numpy(ds.to_frames(xs, np.resize(test_idx, B))).to(dev)
            for _ in range(10):
                soft_readout(model.encode(q)[0][:, 0], zs, sy, model.logtau)
            _sync(dev)
            reps = 50 if B <= 256 else 20
            t0 = time.perf_counter()
            for _ in range(reps):
                soft_readout(model.encode(q)[0][:, 0], zs, sy, model.logtau)
            _sync(dev)
            thr[str(B)] = dict(frames_per_s=B * reps / (time.perf_counter() - t0))
    mem = {}
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        with torch.no_grad():
            for i in test_idx[:50]:
                end_to_end(int(i))
        mem = dict(peak_alloc_MiB=torch.cuda.max_memory_allocated(dev) / 2 ** 20)
    n_anchor = int(len(sidx))
    return dict(info=info,
                model=dict(params=n_params, params_by_module=by_module, size_fp32_MiB=n_params * 4 / 2 ** 20),
                compute=dict(mac_encode=mac_enc, n_anchor=n_anchor, embed_dim=C.FUSE_OUT, mac_anchor_matching=n_anchor * C.FUSE_OUT,
                             anchor_dict_fp32_KiB=n_anchor * (C.FUSE_OUT + 1) * 4 / 1024),
                latency_batch1_ms=e2e, latency_batch1_compute_only_ms=compute_only, throughput=thr,
                feature_step_cpu_us=feature_step_timer(args.fold, goodch, args.data_dir), gpu_memory=mem,
                acquisition_period_ms=C.SAMPLE_PERIOD_MS, fold=args.fold, dataset_no=C.DATASET_NO[args.fold])


def main(argv=None):
    ap = argparse.ArgumentParser(description="model size and batch-1 latency of the decoder")
    ap.add_argument("--out", required=True, help="output json file")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    ap.add_argument("--threads", type=int, default=0, help="PyTorch CPU threads (0 = leave unchanged)")
    ap.add_argument("--iters", type=int, default=1000, help="timed queries after 200 warm-up queries")
    ap.add_argument("--data-dir", default=C.DATA_DIR)
    ap.add_argument("--model-dir", default=os.path.join(C.RAW_DIR, "decoder"))
    ap.add_argument("--fold", default=C.ORDER[0], help="dataset whose weights and test frames are used")
    args = ap.parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_determinism(C.SEED)
    print("command: " + "python -m hingesense.cost " + " ".join(sys.argv[1:]), flush=True)
    res = bench_decoder(args)
    res["command"] = "python -m hingesense.cost " + " ".join(sys.argv[1:])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
