# Fig. S8.4

**Paper item / 文章条目.** Fig. S8.4 of the supplementary material.  
补充材料 Fig. S8.4。

**How to produce it / 生成方法.**

```bash
bash scripts/fig_S8_4.sh
# step by step:
python check_data.py
python train.py --jobs 24 --gpus 0,1
python electrode_importance.py --jobs 24 --gpus 0,1
python make_tables.py --only Fig_S8_4
```

**Inputs / 输入.** `results/raw/decoder/<dataset>_model.pt` and `results/raw/importance/<dataset>.json` (keys `importance`, `importance_norm`). `electrode_ranking.txt` is written by `electrode_importance.py` and read by `train.py --electrodes top4`, `electrode_failure.py` and `make_tables.py`.  
`results/raw/decoder/<数据集>_model.pt` 与 `results/raw/importance/<数据集>.json`（键 `importance`、`importance_norm`）。`electrode_ranking.txt` 由 `electrode_importance.py` 写出，被 `train.py --electrodes top4`、`electrode_failure.py` 与 `make_tables.py` 读取。

**Files / 文件.** `fig_s8_4.png`, `fig_s8_4.pdf`, `electrode_ranking.txt`, `fig_s8_4.csv` (per-electrode values), `fig_s8_4_unrounded.csv` (also per dataset), `provenance.txt`.  
`fig_s8_4.png`、`fig_s8_4.pdf`、`electrode_ranking.txt`、`fig_s8_4.csv`（各电极数值）、`fig_s8_4_unrounded.csv`（另含逐数据集值）、`provenance.txt`。
