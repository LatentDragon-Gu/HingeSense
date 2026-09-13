# Table II

**Paper item / 文章条目.** Table II of the main text.  
正文 Table II。

**How to produce it / 生成方法.**

```bash
bash scripts/table_II.sh
# step by step:
python check_data.py
python train.py --jobs 24 --gpus 0,1
python train.py --method all --jobs 24 --gpus 0,1
python make_tables.py --only Table_II
```

**Inputs / 输入.** `results/raw/decoder/<dataset>.json` (key `mae_test_exact`) and `results/raw/baselines/<method>/<dataset>.json` (keys `mae_seq_exact`, `mae_shuf_exact`). `--augmented` adds `gru_aug`, `tcn_aug`, `rf_aug` when their results exist.  
`results/raw/decoder/<数据集>.json`（键 `mae_test_exact`）与 `results/raw/baselines/<方法>/<数据集>.json`（键 `mae_seq_exact`、`mae_shuf_exact`）。`--augmented` 在结果存在时另加 `gru_aug`、`tcn_aug`、`rf_aug`。

**Files / 文件.** `table_ii.csv` (as printed), `table_ii_unrounded.csv`, `provenance.txt`.  
`table_ii.csv`（文章版式）、`table_ii_unrounded.csv`、`provenance.txt`。
