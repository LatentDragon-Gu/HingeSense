# -*- coding: utf-8 -*-
"""原始结果（results/raw）→ 各表：终端打印（与文章同版式、同精度）并写 csv。由根目录 make_tables.py 调用。

    table_ii_s81       Table II（各方法的均值 / 最差 / 亚度计数）与 Table S8.1（逐数据集 MAE、扫描段数 / 步长行）
    table_iii_s82      Table III（代表通道线性 / 三次读出与迟滞的均值）与 Table S8.2（逐数据集）
    table_iv_s83       Table IV（消融汇总）与 Table S8.3（逐数据集）
    table_s84          Table S8.4（16 电极与排名前 4 电极配置的基准与单电极失效）
    fig_s84_summary    Fig. S8.4 的数值：各电极在两种条件下的重要性均值 / 标准差
    table_s23          Table S2.3（批 1 时延）与 S2.3 正文引用的模型体量数字
每张表两个文件：<表>.csv（文章版式的取整值）、<表>_unrounded.csv（未取整值）。
缺失的输入不会中断：单元格记作 —，函数返回缺失清单与生成命令，由 make_tables.py 打印。

Raw results (results/raw) → tables: printed in the terminal in the layout and precision of the paper and written as csv.
Called by make_tables.py. table_ii_s81 (Table II and S8.1), table_iii_s82 (Table III and S8.2), table_iv_s83 (Table IV and S8.3),
table_s84 (Table S8.4), fig_s84_summary (numbers of Fig. S8.4), table_s23 (Table S2.3 and the model-size numbers of Section S2.3).
Each table gives two files: <table>.csv (rounded values as printed) and <table>_unrounded.csv. Missing inputs do not stop the
build: the cell shows — and the function returns the missing files and the commands that produce them.
"""
import csv
import glob
import json
import os
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

from . import config as C
from . import ranking as R

METHOD_ROWS = [("Linear", "ridge"), ("Exponential Smoothing", "es"), ("Kalman Filter", "kalman"), ("Random Forest", "rf"),
               ("Frame-wise MLP", "mlp"), ("TCN", "tcn"), ("GRU", "gru"), ("Transformer", "transformer"), ("KalmanNet", "kalmannet"),
               ("Calibrated Ridge", "calib_ridge"), ("Calib. Ridge + Kalman", "calib_kalman"), ("Calib. Ridge + ES", "calib_es"),
               ("Soft-kNN (w/o learning)", "rawknn")]
METHOD_ROWS_AUG = [("GRU + aug", "gru_aug"), ("TCN + aug", "tcn_aug"), ("Random Forest + aug", "rf_aug")]
TEMPORAL = ("es", "kalman", "tcn", "gru", "transformer", "kalmannet", "calib_kalman", "calib_es", "gru_aug", "tcn_aug")
NO = [C.DATASET_NO[f] for f in C.ORDER]
HINT_DECODER = "python train.py"
HINT_BASELINES = "python train.py --method all"
HINT_ABLATION = "python train.py --ablation %s"
HINT_TOP4 = "python train.py --electrodes top4"
HINT_IMPORTANCE = "python electrode_importance.py"
HINT_FAILURE = "python electrode_failure.py"
HINT_COST = "python inference_cost.py"


# ====================================================================== 公共工具 / helpers
def rnd(v, nd):
    """半进位取整（十进制），作用于未取整值。 / Decimal half-up rounding of an unrounded value."""
    if v is None:
        return None
    return float(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def fmt(v, nd):
    return "—" if v is None else ("%%.%df" % nd) % rnd(v, nd)


def _agg(vals):
    """均值、最差值、亚度（<1°）数据集个数；任一缺失则返回 None。 / Mean, worst and number of sub-degree (<1°) datasets; None if any value is missing."""
    if not all(v is not None for v in vals):
        return None
    return float(np.mean(vals)), float(max(vals)), int(sum(v < 1 for v in vals))


def _jload(path):
    d = json.load(open(path, encoding="utf-8"))
    return d[0] if isinstance(d, list) else d


def _dedupe(seq):
    return list(dict.fromkeys(seq))


def render(header, rows):
    """对齐的纯文本表：首列左对齐，其余右对齐。 / Aligned plain-text table: first column left-aligned, the others right-aligned."""
    cells = [[str(c) for c in header]] + [[str(c) for c in r] for r in rows]
    n = len(header)
    w = [max(len(r[i]) for r in cells) for i in range(n)]

    def line(r):
        return "  ".join((r[i].ljust(w[i]) if i == 0 else r[i].rjust(w[i])) for i in range(n)).rstrip()

    return "\n".join([line(cells[0]), "  ".join("-" * x for x in w)] + [line(r) for r in cells[1:]])


class Table:
    """一张表：名称（结果文件夹）、文件名、标题、表头、文章版式的行、未取整的行、缺失输入。 / One table: folder name, file stem, title, header, rows as printed, unrounded rows, missing inputs."""

    def __init__(self, name, stem, title, header, out_root, raw_header=None):
        self.name, self.stem, self.title, self.header, self.out_root = name, stem, title, header, out_root
        self.raw_header = raw_header or header
        self.rows, self.raw, self.missing = [], [], []

    def value(self, path, key):
        """读取一个 json 指标（优先 <key>_exact）；文件缺失时登记缺失并返回 None。 / Read one json metric (<key>_exact preferred); a missing file is recorded and gives None."""
        if not os.path.exists(path):
            self.missing.append(os.path.relpath(path, C.ROOT_DIR))
            return None
        d = _jload(path)
        return d.get(key + "_exact" if key + "_exact" in d else key)

    def add(self, row, raw_row=None):
        self.rows.append(row)
        if raw_row is not None:
            self.raw.append(raw_row)

    def text(self):
        return self.title + "\n" + render(self.header, self.rows)

    def write(self):
        d = os.path.join(self.out_root, self.name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, self.stem + ".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.header)
            w.writerows(self.rows)
        with open(os.path.join(d, self.stem + "_unrounded.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.raw_header)
            w.writerows([["" if v is None else v for v in r] for r in self.raw])
        return [os.path.join(d, self.stem + ".csv"), os.path.join(d, self.stem + "_unrounded.csv")]


# ====================================================================== Table II / S8.1
def cycles_and_intervals(data_dir=C.DATA_DIR):
    """每个被测集的扫描段数（周期编号计数）与相邻样本角度步长（非零 |Δθ| 的中位数）。 / Scan-segment count (distinct cycle indices) and angular step (median non-zero |Δθ|) of every evaluated dataset."""
    import pandas as pd
    from .dataset import find_parquet
    out = []
    for f in C.ORDER:
        p = find_parquet(data_dir, f)
        df = pd.read_parquet(p, columns=[C.SEGMENT_COLUMN, C.ANGLE_COLUMN])
        seg = df[C.SEGMENT_COLUMN].to_numpy()
        y = df[C.ANGLE_COLUMN].to_numpy(np.float64)
        d = np.abs(np.diff(y))
        out.append((len(dict.fromkeys(seg.tolist())), float(np.median(d[d > 1e-6]))))
    return out


def table_ii_s81(raw, out_root, data_dir=C.DATA_DIR, augmented=False):
    t2 = Table("Table_II", "table_ii", "Table II. Aggregate MAE (°) comparison over 12 datasets (sequential / block-shuffled for temporal methods)",
               ["Method", "Mean", "Worst", "Sub-degree datasets"], out_root, raw_header=["method", "mean", "worst", "sub_degree_datasets"])
    s1 = Table("Table_S8_1", "table_s8_1", "Table S8.1. MAE (°) comparison on 12 datasets (sequential / block-shuffled for temporal methods)",
               ["No."] + [str(n) for n in NO] + ["Mean"], out_root, raw_header=["method"] + [str(n) for n in NO] + ["mean"])
    ci = cycles_and_intervals(data_dir)
    s1.add(["Scan segments / step (°)"] + ["%d/%.2f" % (c[0], c[1]) for c in ci] + ["—"])
    s1.raw.append(["scan_segments"] + [c[0] for c in ci] + [None])
    s1.raw.append(["step_deg"] + [c[1] for c in ci] + [None])
    hints = []
    for label, m in METHOD_ROWS + (METHOD_ROWS_AUG if augmented else []):
        seq = [s1.value(os.path.join(raw, "baselines", m, f + ".json"), "mae_seq") for f in C.ORDER]
        temporal = m in TEMPORAL
        shuf = [s1.value(os.path.join(raw, "baselines", m, f + ".json"), "mae_shuf") for f in C.ORDER] if temporal else None
        a, b = _agg(seq), (_agg(shuf) if temporal else None)
        cells = [fmt(seq[i], 1) + ("/" + fmt(shuf[i], 1) if temporal else "") for i in range(12)]
        mean_s = fmt(a[0], 2) if a else "—"
        worst_s = fmt(a[1], 1) if a else "—"
        sub_s = ("%d/12" % a[2]) if a else "—"
        if temporal:
            mean_s += "/" + (fmt(b[0], 2) if b else "—")
            worst_s += "/" + (fmt(b[1], 1) if b else "—")
            sub_s += " / " + (("%d/12" % b[2]) if b else "—")
        s1.add([label] + cells + [mean_s], [m + "_sequential"] + seq + [a[0] if a else None])
        t2.add([label, mean_s, worst_s, sub_s], [m + "_sequential"] + (list(a) if a else [None] * 3))
        if temporal:
            s1.raw.append([m + "_block_shuffled"] + shuf + [b[0] if b else None])
            t2.raw.append([m + "_block_shuffled"] + (list(b) if b else [None] * 3))
        if not a or (temporal and not b):
            hints.append(HINT_BASELINES if m in C.PAPER_METHODS else "python train.py --method %s" % m)
    ours = [s1.value(os.path.join(raw, "decoder", f + ".json"), "mae_test") for f in C.ORDER]
    a = _agg(ours)
    s1.add(["Ours"] + [fmt(v, 3) for v in ours] + [fmt(a[0], 3) if a else "—"], ["ours"] + ours + [a[0] if a else None])
    t2.add(["Ours", fmt(a[0], 3) if a else "—", fmt(a[1], 3) if a else "—", ("%d/12" % a[2]) if a else "—"], ["ours"] + (list(a) if a else [None] * 3))
    if not a:
        hints.append(HINT_DECODER)
    t2.missing = list(s1.missing)
    return dict(tables=[t2, s1], missing=_dedupe(s1.missing), hints=_dedupe(hints))


# ====================================================================== Table III / S8.2
def direction_sign(y):
    d = np.sign(np.diff(y, prepend=y[0]))
    last = 0.0
    for k in range(len(d)):
        if d[k] != 0:
            last = d[k]
        else:
            d[k] = last
    return d


def loop_width(yy, val, up, dn, min_pts=8, min_span=10.0):
    """同一周期内升 / 降两支在公共 1° 网格上的平均绝对间隔。 / Mean absolute gap between the ascending and descending branches of one cycle on a common 1° grid."""
    if up.sum() < min_pts or dn.sum() < min_pts:
        return None
    yu, vu = yy[up], val[up]
    yd, vd = yy[dn], val[dn]
    ou, od = np.argsort(yu), np.argsort(yd)
    yu, vu, yd, vd = yu[ou], vu[ou], yd[od], vd[od]
    lo, hi = max(yu.min(), yd.min()), min(yu.max(), yd.max())
    if hi - lo < min_span:
        return None
    g = np.arange(lo, hi, 1.0)
    return float(np.mean(np.abs(np.interp(g, yu, vu) - np.interp(g, yd, vd))))


def representative_channel_benchmark(raw, data_dir=C.DATA_DIR):
    """代表通道基准：候选电极对 E3–E12 与 E4–E11（1 基编号 4–13 与 5–12），在验证段上按与角度的相关性选通道；
    线性 / 三次映射按运动方向在验证段拟合、在测试段评估；迟滞按测试段逐周期计算后取平均。本方法用其保存的测试段预测。
    Representative-channel benchmark: candidate pairs E3–E12 and E4–E11 (one-based 4–13 and 5–12), chosen per dataset by the
    correlation with the angle on the validation segment; linear / cubic maps fitted per motion direction on the validation
    segment and evaluated on the test segment; hysteresis per test cycle, averaged. The decoder uses its saved test predictions."""
    import pandas as pd
    from .dataset import find_parquet, load_raw
    cands = {"4–13": C.channel_index(3, 12), "5–12": C.channel_index(4, 11)}
    rows = []
    for f in C.ORDER:
        z = load_raw(f, data_dir)
        y = z["y"].astype(np.float64)
        x = z["x_clean"].astype(np.float64)
        v0, t0 = int(z["q0"]), int(z["t0"])
        cors = {k: abs(np.corrcoef(x[v0:t0, ch], y[v0:t0])[0, 1]) for k, ch in cands.items()}
        pair = "4–13" if cors["4–13"] >= cors["5–12"] else "5–12"
        ch = cands[pair]
        d = direction_sign(y)
        yv, rv, dv = y[v0:t0], x[v0:t0, ch], d[v0:t0]
        yt, rt, dt = y[t0:], x[t0:, ch], d[t0:]
        up, dn = dt > 0, dt < 0

        def fit_eval(deg):
            out = []
            for mv, mt in ((dv > 0, up), (dv < 0, dn)):
                c = np.polyfit(rv[mv], yv[mv], deg)
                out.append(float(np.mean(np.abs(np.polyval(c, rt[mt]) - yt[mt]))))
            return out

        lu, ld = fit_eval(1)
        cu, cd = fit_eval(3)
        lin_pred = np.polyval(np.polyfit(rv, yv, 1), rt)
        pred_path = os.path.join(raw, "decoder", f + ".npy")
        pred = np.load(pred_path).astype(np.float64) if os.path.exists(pred_path) else None
        seg = pd.read_parquet(find_parquet(data_dir, f), columns=[C.SEGMENT_COLUMN])[C.SEGMENT_COLUMN].to_numpy()[t0:]

        def cycle_hyst(val):
            ws = []
            for c in dict.fromkeys(seg.tolist()):
                m = seg == c
                w = loop_width(yt[m], val[m], up[m], dn[m])
                if w is not None:
                    ws.append(w)
            return float(np.mean(ws)) if ws else None

        ou = float(np.mean(np.abs(pred[up] - yt[up]))) if pred is not None else None
        od = float(np.mean(np.abs(pred[dn] - yt[dn]))) if pred is not None else None
        rows.append(dict(no=C.DATASET_NO[f], fold=f, pair=pair, channel=int(ch), lin_up=lu, lin_dn=ld, cub_up=cu, cub_dn=cd, sc_hyst=cycle_hyst(lin_pred),
                         ours_up=ou, ours_dn=od, ours_hyst=(cycle_hyst(pred) if pred is not None else None),
                         pred=(os.path.relpath(pred_path, C.ROOT_DIR) if pred is not None else None)))
    return rows


def table_iii_s82(raw, out_root, data_dir=C.DATA_DIR):
    rows = representative_channel_benchmark(raw, data_dir)
    keys = [("lin_up", 2), ("lin_dn", 2), ("cub_up", 2), ("cub_dn", 2), ("sc_hyst", 2), ("ours_up", 3), ("ours_dn", 3), ("ours_hyst", 3)]
    s2 = Table("Table_S8_2", "table_s8_2", "Table S8.2. Direction-resolved single-channel generalization and hysteresis analysis (°)",
               ["No.", "Electrode pair", "Linear ↑", "Linear ↓", "Cubic ↑", "Cubic ↓", "Single-channel hysteresis", "Ours ↑", "Ours ↓", "Ours hysteresis"], out_root,
               raw_header=["no", "electrode_pair", "channel"] + [k for k, _ in keys])
    for r in rows:
        s2.add([r["no"], r["pair"]] + [fmt(r[k], nd) for k, nd in keys], [r["no"], r["pair"], r["channel"]] + [r[k] for k, _ in keys])
        if r["pred"] is None:
            s2.missing.append(os.path.join("results", "raw", "decoder", r["fold"] + ".npy"))
    means = {k: (float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None) for k, _ in keys}
    s2.add(["Mean", "—"] + [fmt(means[k], nd) for k, nd in keys], ["mean", None, None] + [means[k] for k, _ in keys])
    t3 = Table("Table_III", "table_iii", "Table III. Representative-channel generalization benchmark and cycle-wise hysteresis (12-dataset mean, °)",
               ["Readout", "MAE ↑", "MAE ↓", "Cycle-wise hysteresis"], out_root, raw_header=["readout", "mae_ascending", "mae_descending", "cycle_wise_hysteresis"])
    t3.add(["Representative single-channel linear", fmt(means["lin_up"], 2), fmt(means["lin_dn"], 2), fmt(means["sc_hyst"], 2)], ["linear", means["lin_up"], means["lin_dn"], means["sc_hyst"]])
    t3.add(["Representative single-channel cubic", fmt(means["cub_up"], 2), fmt(means["cub_dn"], 2), "—"], ["cubic", means["cub_up"], means["cub_dn"], None])
    t3.add(["Proposed multi-electrode decoder", fmt(means["ours_up"], 3), fmt(means["ours_dn"], 3), fmt(means["ours_hyst"], 3)], ["ours", means["ours_up"], means["ours_dn"], means["ours_hyst"]])
    t3.missing = list(s2.missing)
    miss = _dedupe(s2.missing)
    return dict(tables=[t3, s2], missing=miss, hints=[HINT_DECODER] if miss else [])


# ====================================================================== Table IV / S8.3
def table_iv_s83(raw, out_root):
    t4 = Table("Table_IV", "table_iv", "Table IV. Ablation analysis under the 12-dataset leave-one-dataset-out protocol (MAE, °)",
               ["Configuration", "Mean", "Worst", "Sub-deg. datasets"], out_root, raw_header=["configuration", "mean", "worst", "sub_degree_datasets"])
    s3 = Table("Table_S8_3", "table_s8_3", "Table S8.3. Dataset-wise ablation results (MAE, °)",
               ["Configuration"] + [str(n) for n in NO] + ["Mean"], out_root, raw_header=["configuration"] + [str(n) for n in NO] + ["mean"])
    hints = []
    for name in ["decoder"] + list(C.ABLATIONS):
        label = C.ABLATION_LABELS[name]
        sub = "decoder" if name == "decoder" else "ablation_" + name
        vals = [s3.value(os.path.join(raw, sub, f + ".json"), "mae_test") for f in C.ORDER]
        a = _agg(vals)
        s3.add([label] + [fmt(v, 3) for v in vals] + [fmt(a[0], 3) if a else "—"], [name] + vals + [a[0] if a else None])
        t4.add([label, fmt(a[0], 3) if a else "—", fmt(a[1], 2) if a else "—", ("%d/12" % a[2]) if a else "—"], [name] + (list(a) if a else [None] * 3))
        if not a:
            hints.append(HINT_DECODER if name == "decoder" else HINT_ABLATION % name)
    t4.missing = list(s3.missing)
    return dict(tables=[t4, s3], missing=_dedupe(s3.missing), hints=_dedupe(hints))


# ====================================================================== Table S8.4
def table_s84(raw, out_root, ranking_path=C.RANKING_FILE):
    s4 = Table("Table_S8_4", "table_s8_4", "Table S8.4. Electrode reduction and electrode-failure results (MAE, °)",
               ["Configuration"] + [str(n) for n in NO] + ["Mean"], out_root, raw_header=["configuration"] + [str(n) for n in NO] + ["mean"])
    hints = []
    try:
        subset = R.top_electrodes(C.SUBSET_ELECTRODES, ranking_path)
        failures = R.top_electrodes(C.FAILURE_ELECTRODES, ranking_path)
    except FileNotFoundError:
        s4.missing.append(os.path.relpath(ranking_path, C.ROOT_DIR))
        return dict(tables=[s4], missing=s4.missing, hints=[HINT_IMPORTANCE])
    configs = [("16-electrode baseline", "decoder", "16 electrodes"),
               ("%d-electrode baseline" % len(subset), "electrodes_top4", "%d electrodes" % len(subset))]
    for base_label, sub, short in configs:
        vals = [s4.value(os.path.join(raw, sub, f + ".json"), "mae_test") for f in C.ORDER]
        a = _agg(vals)
        s4.add([base_label] + [fmt(v, 3) for v in vals] + [fmt(a[0], 3) if a else "—"], [sub] + vals + [a[0] if a else None])
        if not a:
            hints.append(HINT_DECODER if sub == "decoder" else HINT_TOP4)
        for e in failures:
            label = "%s with E%d failure" % (short, e)
            vals = [s4.value(os.path.join(raw, "electrode_failure", sub, "%s_E%d.json" % (f, e)), "mae_failure") for f in C.ORDER]
            a = _agg(vals)
            s4.add([label] + [fmt(v, 3) for v in vals] + [fmt(a[0], 3) if a else "—"], ["%s_E%d_failure" % (sub, e)] + vals + [a[0] if a else None])
            if not a:
                hints.append(HINT_FAILURE)
    return dict(tables=[s4], missing=_dedupe(s4.missing), hints=_dedupe(hints))


# ====================================================================== Fig. S8.4 数值 / numbers of Fig. S8.4
def importance_by_fold(raw):
    """{折名: 16 维归一化重要性}，只含已有结果的折。 / {dataset: 16 normalised importances} for the datasets that have results."""
    M = {}
    for f in C.ORDER:
        p = os.path.join(raw, "importance", f + ".json")
        if os.path.exists(p):
            d = _jload(p)["importance_norm"]
            M[f] = [float(d["e%d" % e]) for e in range(C.N_ELECTRODE)]
    return M


def fig_s84_summary(raw, out_root, ranking_path=C.RANKING_FILE):
    M = importance_by_fold(raw)
    tb = Table("Fig_S8_4", "fig_s8_4", "Fig. S8.4. Normalized electrode permutation importance: mean / std over the datasets of each condition, global mean and rank",
               ["Electrode", "No-load mean", "No-load std", "Assembly mean", "Assembly std", "Global mean", "Rank"], out_root,
               raw_header=["electrode", "no_load_mean", "no_load_std", "assembly_mean", "assembly_std", "global_mean", "rank"] + ["dataset_%d" % n for n in NO])
    missing = [os.path.relpath(os.path.join(raw, "importance", f + ".json"), C.ROOT_DIR) for f in C.ORDER if f not in M]
    if missing:
        tb.missing += missing
        return dict(tables=[tb], missing=missing, hints=[HINT_IMPORTANCE], norm_by_fold=M)
    noload = np.array([M[f] for f in C.ORDER if C.cond_of(f) != "load"])
    assembly = np.array([M[f] for f in C.ORDER if C.cond_of(f) == "load"])
    G = np.array([M[f] for f in C.ORDER]).mean(0)
    try:
        rank = {r["electrode"]: r["rank"] for r in R.load_ranking(ranking_path)}
    except FileNotFoundError:
        rank = {}
    for e in range(C.N_ELECTRODE):
        row = [float(noload.mean(0)[e]), float(noload.std(0)[e]), float(assembly.mean(0)[e]), float(assembly.std(0)[e]), float(G[e])]
        tb.add(["E%d" % e] + [fmt(v, 3) for v in row] + [rank.get(e, "—")], ["E%d" % e] + row + [rank.get(e, None)] + [M[f][e] for f in C.ORDER])
    return dict(tables=[tb], missing=[], hints=[] if rank else [HINT_IMPORTANCE], norm_by_fold=M)


# ====================================================================== Table S2.3
def table_s23(raw, out_root):
    files = sorted(glob.glob(os.path.join(raw, "inference_cost", "decoder_*.json")), key=lambda p: (0 if "cpu" in os.path.basename(p) else 1, p))
    tb = Table("Table_S2_3", "table_s2_3", "Table S2.3. Batch-size-one decoder latency", ["Platform", "PyTorch", "Median (ms)", "P95 (ms)"], out_root,
               raw_header=["file", "platform", "pytorch", "median_ms", "p95_ms", "mean_ms", "p99_ms", "timed_queries", "anchors"])
    ms = Table("Table_S2_3", "model_size", "Section S2.3. Model size and cost of the decoder", ["Quantity", "Value"], out_root, raw_header=["quantity", "value"])
    if not files:
        tb.missing.append(os.path.relpath(os.path.join(raw, "inference_cost"), C.ROOT_DIR))
        return dict(tables=[tb, ms], missing=tb.missing, hints=[HINT_COST])
    for p in files:
        d = _jload(p)
        lat = d["latency_batch1_ms"]
        tb.add([d["info"]["platform"], d["info"]["pytorch"], "%.3f" % lat["median_ms"], "%.3f" % lat["p95_ms"]],
               [os.path.relpath(p, C.ROOT_DIR), d["info"]["platform"], d["info"]["pytorch"], lat["median_ms"], lat["p95_ms"], lat["mean_ms"], lat["p99_ms"], lat["n"], d["compute"]["n_anchor"]])
    cpu = [p for p in files if "cpu" in os.path.basename(p)]
    d = _jload(cpu[0] if cpu else files[0])
    fs = d["feature_step_cpu_us"]
    items = [("Parameters", "%d" % d["model"]["params"], d["model"]["params"]),
             ("Model size in FP32 (MiB)", "%.3f" % d["model"]["size_fp32_MiB"], d["model"]["size_fp32_MiB"]),
             ("Linear-layer MACs per query (encoding)", "%d" % d["compute"]["mac_encode"], d["compute"]["mac_encode"]),
             ("Cached anchors (dataset %d)" % d["dataset_no"], "%d" % d["compute"]["n_anchor"], d["compute"]["n_anchor"]),
             ("Anchor embeddings + angle labels in FP32 (KiB)", "%.1f" % d["compute"]["anchor_dict_fp32_KiB"], d["compute"]["anchor_dict_fp32_KiB"]),
             ("Causal feature update per frame, CPU median (µs)", "%.1f" % fs["median_us"], fs["median_us"])]
    for q, s, v in items:
        ms.add([q, s], [q, v])
    return dict(tables=[tb, ms], missing=[], hints=[])
