# Table S8.1

**Paper item / 文章条目.** Table S8.1 of the supplementary material.  
补充材料 Table S8.1。

**How to produce it / 生成方法.**

```bash
bash scripts/table_S8_1.sh
# same inputs as Table II; or
python make_tables.py --only Table_S8_1
```

**Inputs / 输入.** Same as `../Table_II/`; the first row is computed from the parquet files (number of distinct `xdat_segment` values, median non-zero |Δangle|).  
与 `../Table_II/` 相同；首行由 parquet 文件算出（`xdat_segment` 取值个数、非零 |Δ角度| 的中位数）。

**Files / 文件.** `table_s8_1.csv`, `table_s8_1_unrounded.csv` (one row per method and evaluation mode), `provenance.txt`.  
`table_s8_1.csv`、`table_s8_1_unrounded.csv`（每种方法与评测方式一行）、`provenance.txt`。
