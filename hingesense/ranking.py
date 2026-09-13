# -*- coding: utf-8 -*-
"""电极重要性排名文件（results/Fig_S8_4/electrode_ranking.txt）的写入与解析。

该文件由根目录 electrode_importance.py 在 12 个被测数据集的置换重要性都齐全后生成，是电极精简配置（排名前 4）、
单电极失效测试的电极（排名前 3）与 Fig. S8.4 星标的唯一来源；其他程序只读取解析它，不各自计算。
格式：以 # 开头的说明行，随后每行 "rank electrode score_global score_noload score_assembly"。

Writer and parser of the electrode ranking file (results/Fig_S8_4/electrode_ranking.txt). electrode_importance.py writes it
once the permutation importance of all 12 evaluated datasets exists; it is the only source of the reduced configuration (top 4),
of the electrodes tested for failure (top 3) and of the stars in Fig. S8.4. Format: comment lines starting with #, then one line
"rank electrode score_global score_noload score_assembly" per electrode.
"""
import os

import numpy as np

from . import config as C

COLUMNS = ("rank", "electrode", "score_global", "score_noload", "score_assembly")


def build_ranking(norm_by_fold):
    """{折名: 16 维归一化重要性} → 按全局均值降序的行列表。全局 = 12 个被测集的均值；并列时电极号小者在前。
    {dataset: 16 normalised importances} → rows sorted by the global mean (mean over the 12 datasets); ties keep the lower electrode first."""
    missing = [f for f in C.ORDER if f not in norm_by_fold]
    if missing:
        raise ValueError("importance results missing for %d datasets: %s" % (len(missing), ", ".join(missing)))
    M = np.array([norm_by_fold[f] for f in C.ORDER], dtype=np.float64)
    is_load = np.array([C.cond_of(f) == "load" for f in C.ORDER])
    g, e, l = M.mean(0), M[~is_load].mean(0), M[is_load].mean(0)
    order = np.argsort(-g, kind="stable")
    return [dict(rank=r + 1, electrode=int(k), score_global=float(g[k]), score_noload=float(e[k]), score_assembly=float(l[k]))
            for r, k in enumerate(order)]


def write_ranking(path, rows, command, permutations=C.IMPORTANCE_SEEDS, seed=C.SEED):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# electrode permutation importance, global ranking\n")
        f.write("# score_global = mean of the normalised importance over the %d evaluated datasets; "
                "score_noload = datasets %s; score_assembly = datasets %s\n" % (
                    len(C.ORDER), ",".join(str(C.DATASET_NO[x]) for x in C.ORDER if C.cond_of(x) != "load"),
                    ",".join(str(C.DATASET_NO[x]) for x in C.ORDER if C.cond_of(x) == "load")))
        f.write("# produced by: %s   (seed %d, %d permutations per electrode)\n" % (command, seed, permutations))
        f.write("# " + " ".join(COLUMNS) + "\n")
        for r in rows:
            f.write("%d E%d %.6f %.6f %.6f\n" % (r["rank"], r["electrode"], r["score_global"], r["score_noload"], r["score_assembly"]))


def load_ranking(path=C.RANKING_FILE):
    """读取排名文件 → 按 rank 排序的行列表；文件不存在时给出生成命令。 / Read the ranking file → rows sorted by rank; names the producing command when the file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError("%s not found. Produce it first with:  python electrode_importance.py" % path)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5 or not parts[1].startswith("E"):
                raise ValueError("malformed line in %s: %r" % (path, line))
            rows.append(dict(rank=int(parts[0]), electrode=int(parts[1][1:]), score_global=float(parts[2]),
                             score_noload=float(parts[3]), score_assembly=float(parts[4])))
    rows.sort(key=lambda r: r["rank"])
    if [r["rank"] for r in rows] != list(range(1, C.N_ELECTRODE + 1)):
        raise ValueError("%s must list every electrode exactly once with ranks 1..%d" % (path, C.N_ELECTRODE))
    return rows


def top_electrodes(k, path=C.RANKING_FILE):
    """排名前 k 的电极（升序编号）。 / The k highest-ranked electrodes, sorted by number."""
    return sorted(r["electrode"] for r in load_ranking(path)[:k])


def parse_electrodes(spec, path=C.RANKING_FILE):
    """命令行电极选择："top4" 这类按排名文件取前 k 个；"0,1,8,15" 这类为显式列表。返回升序电极号列表。
    Command-line electrode selection: "topK" takes the top K of the ranking file, "0,1,8,15" is an explicit list; returns sorted electrode numbers."""
    spec = str(spec).strip().lower()
    if spec.startswith("top") and spec[3:].isdigit():
        return top_electrodes(int(spec[3:]), path)
    es = sorted(set(int(t) for t in spec.replace("e", "").split(",") if t.strip()))
    if not es or any(e < 0 or e >= C.N_ELECTRODE for e in es):
        raise ValueError("electrodes must be 'topK' or a comma-separated list within 0..%d" % (C.N_ELECTRODE - 1))
    return es
