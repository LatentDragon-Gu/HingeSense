# Table S2.3

**Paper item / 文章条目.** Table S2.3 of the supplementary material, measured on the machine that ran the script; `model_size.csv` holds the model-size numbers quoted in Section S2.3.  
补充材料 Table S2.3，在运行脚本的机器上测得；`model_size.csv` 为 S2.3 正文引用的模型体量数字。

**How to produce it / 生成方法.**

```bash
bash scripts/table_S2_3.sh
# step by step:
python check_data.py
python train.py --folds 1
python inference_cost.py
python make_tables.py --only Table_S2_3
```

**Inputs / 输入.** `results/raw/inference_cost/decoder_cpu1.json` and `decoder_gpu.json` (keys `info.platform`, `info.pytorch`, `latency_batch1_ms`, `model`, `compute`, `feature_step_cpu_us`).  
`results/raw/inference_cost/decoder_cpu1.json` 与 `decoder_gpu.json`（键 `info.platform`、`info.pytorch`、`latency_batch1_ms`、`model`、`compute`、`feature_step_cpu_us`）。

**Files / 文件.** `table_s2_3.csv`, `table_s2_3_unrounded.csv`, `model_size.csv`, `model_size_unrounded.csv`, `provenance.txt`.  
`table_s2_3.csv`、`table_s2_3_unrounded.csv`、`model_size.csv`、`model_size_unrounded.csv`、`provenance.txt`。
