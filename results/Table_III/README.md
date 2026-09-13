# Table III

**Paper item / 文章条目.** Table III of the main text.  
正文 Table III。

**How to produce it / 生成方法.**

```bash
bash scripts/table_III.sh
# step by step:
python check_data.py
python train.py --jobs 24 --gpus 0,1
python make_tables.py --only Table_III
```

**Inputs / 输入.** The parquet files (single-channel columns) and `results/raw/decoder/<dataset>.npy` (decoder columns).  
parquet 文件（单通道各列）与 `results/raw/decoder/<数据集>.npy`（解码器各列）。

**Files / 文件.** `table_iii.csv`, `table_iii_unrounded.csv`, `provenance.txt`.  
`table_iii.csv`、`table_iii_unrounded.csv`、`provenance.txt`。
