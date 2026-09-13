# Table IV

**Paper item / 文章条目.** Table IV of the main text.  
正文 Table IV。

**How to produce it / 生成方法.**

```bash
bash scripts/table_IV.sh
# step by step:
python check_data.py
python train.py --jobs 24 --gpus 0,1
for a in no_drift_correction no_causal_differential no_channel_mask no_reference_pathway \
         no_noise_perturbation no_symmetry_augmentation no_augmentation; do
  python train.py --ablation $a --jobs 24 --gpus 0,1
done
python make_tables.py --only Table_IV
```

**Inputs / 输入.** `results/raw/decoder/<dataset>.json` and `results/raw/ablation_<name>/<dataset>.json` (key `mae_test_exact`).  
`results/raw/decoder/<数据集>.json` 与 `results/raw/ablation_<名称>/<数据集>.json`（键 `mae_test_exact`）。

**Files / 文件.** `table_iv.csv`, `table_iv_unrounded.csv`, `provenance.txt`.  
`table_iv.csv`、`table_iv_unrounded.csv`、`provenance.txt`。
