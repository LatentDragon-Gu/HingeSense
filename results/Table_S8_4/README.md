# Table S8.4

**Paper item / 文章条目.** Table S8.4 of the supplementary material.  
补充材料 Table S8.4。

**How to produce it / 生成方法.**

```bash
bash scripts/table_S8_4.sh
# step by step:
python check_data.py
python train.py --jobs 24 --gpus 0,1
python electrode_importance.py --jobs 24 --gpus 0,1
python train.py --electrodes top4 --jobs 24 --gpus 0,1
python electrode_failure.py --jobs 24 --gpus 0,1
python make_tables.py --only Table_S8_4
```

**Inputs / 输入.** `results/raw/decoder/<dataset>.json`, `results/raw/electrodes_top4/<dataset>.json` (key `mae_test_exact`), `results/raw/electrode_failure/<configuration>/<dataset>_E<k>.json` (key `mae_failure_exact`), and `../Fig_S8_4/electrode_ranking.txt` (which electrodes).  
`results/raw/decoder/<数据集>.json`、`results/raw/electrodes_top4/<数据集>.json`（键 `mae_test_exact`）、`results/raw/electrode_failure/<配置>/<数据集>_E<k>.json`（键 `mae_failure_exact`），以及 `../Fig_S8_4/electrode_ranking.txt`（决定电极）。

**Files / 文件.** `table_s8_4.csv`, `table_s8_4_unrounded.csv`, `provenance.txt`.  
`table_s8_4.csv`、`table_s8_4_unrounded.csv`、`provenance.txt`。
