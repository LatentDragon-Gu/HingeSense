# -*- coding: utf-8 -*-
"""由结果直接生成的图。

    fig_importance   Fig. S8.4：电极置换重要性的雷达图（(a) 无载、(b) 装配），星标电极读自排名文件
由根目录 make_tables.py 调用。

Figures drawn from the results. fig_importance: Fig. S8.4, radar plots of the electrode permutation importance
((a) no-load, (b) assembly); the starred electrodes come from the ranking file. Called by make_tables.py.
"""
import os

import numpy as np

from . import config as C
from . import ranking as R


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "axes.linewidth": 0.8})
    return plt


def _save(fig, path_no_ext):
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    fig.savefig(path_no_ext + ".png", bbox_inches="tight", dpi=600)
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    return [path_no_ext + ".png", path_no_ext + ".pdf"]


def fig_importance(norm_by_fold, out_dir, ranking_path=C.RANKING_FILE, n_star=C.SUBSET_ELECTRODES):
    """norm_by_fold = {折名: 16 维归一化重要性}（12 个被测集齐全）。星标 = 排名文件前 n_star 个电极。 / norm_by_fold = {dataset: 16 normalised importances} for all 12 datasets; stars = top n_star electrodes of the ranking file."""
    plt = _plt()
    stars = R.top_electrodes(n_star, ranking_path)
    star_text = ", ".join("E%d" % e for e in stars)
    noload = [f for f in C.ORDER if C.cond_of(f) != "load"]
    assembly = [f for f in C.ORDER if C.cond_of(f) == "load"]
    groups = [("No-load (datasets %d-%d)" % (C.DATASET_NO[noload[0]], C.DATASET_NO[noload[-1]]), noload),
              ("Assembly (datasets %d-%d)" % (C.DATASET_NO[assembly[0]], C.DATASET_NO[assembly[-1]]), assembly)]
    ang = np.linspace(0, 2 * np.pi, C.N_ELECTRODE, endpoint=False)
    close = lambda v: np.concatenate([v, v[:1]])
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.15), dpi=300, subplot_kw=dict(polar=True))
    for ax, (title, folds) in zip(axes, groups):
        for i, f in enumerate(folds):
            ax.plot(close(ang), close(np.asarray(norm_by_fold[f])), color=cmap(i), lw=0.9, alpha=0.75, label="No.%d" % C.DATASET_NO[f], zorder=4)
        m = np.mean([norm_by_fold[f] for f in folds], axis=0)
        ax.plot(close(ang), close(m), color="black", lw=2.0, label="mean", zorder=6)
        ax.plot(close(ang), close(m), "o", color="black", ms=2.4, zorder=7)
        for e in stars:
            ax.plot([ang[e]], [1.02], marker="*", color="#333333", ms=7, clip_on=False, zorder=8)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(ang)
        ax.set_xticklabels(["E%d" % i for i in range(C.N_ELECTRODE)], fontsize=8)
        ax.set_rlim(0, 1.05)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.5)
        ax.set_rlabel_position(56)
        ax.tick_params(pad=1.5)
        ax.grid(lw=0.5, alpha=0.55)
        ax.spines["polar"].set_alpha(0.4)
        ax.set_title(title, fontsize=10, pad=14)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=4, fontsize=7, frameon=False, handlelength=1.4, columnspacing=0.9, handletextpad=0.5)
    fig.text(0.5, 0.012, r"$\bigstar$ globally top-%d electrodes (%s)" % (n_star, star_text), ha="center", fontsize=7, color="#333333")
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    files = _save(fig, os.path.join(out_dir, "fig_s8_4"))
    plt.close(fig)
    return files
