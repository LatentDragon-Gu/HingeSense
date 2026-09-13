# HingeSense

[中文说明 / Chinese version: README_cn.md](README_cn.md)

Code and data of the paper

> P. Gu, H. Zhang, S. Qu, *A Geometry-Preserving Self-Sensing SCM Hinge with Calibration-Anchored Decoding for
> Sub-Degree Joint Proprioception*, IEEE Transactions on Industrial Electronics.

The repository produces Tables II, III and IV of the main text, Tables S8.1, S8.2, S8.3 and S8.4 and Fig. S8.4 of the
supplementary material, and Table S2.3 for the machine it runs on. Each of them has one shell script in `scripts/`;
the script prints the table in the terminal and writes it as csv under `results/`. `results/` already holds the output
of one complete run, including the trained weights and the test-segment predictions, so the tables can also be rebuilt
without training.

## 1. Quick start

```bash
conda create -n hingesense python=3.11 && conda activate hingesense
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130   # choose the index of your CUDA version
pip install -r requirements.txt
JOBS=24 GPUS=0,1 bash scripts/table_IV.sh                                        # trains and prints Table IV
```

Each script first verifies the data files (about ten seconds), then runs the training or evaluation commands that its
table needs, then prints the table and writes it to `results/<table>/`. Results that already exist under `results/raw/`
are reused, so a script can be interrupted and started again, and scripts that share inputs (all of them share the
decoder training) do not repeat work. To rebuild every table from the shipped raw results without any training:

```bash
python make_tables.py
```

Three environment variables control the scripts: `JOBS` (number of parallel processes, default 1), `GPUS` (GPU ids to
rotate over, e.g. `0,1`; default all visible GPUs) and `PYTHON` (interpreter, default `python`). On Windows without
bash, run the python commands listed in Section 3 one after another.

## 2. Layout

```
HingeSense/
├── README.md, README_cn.md    this file, and its Chinese version
├── LICENSE                    MIT (code); Data/LICENSE and Supplementary/LICENSE are CC BY 4.0 (data)
├── CITATION.cff
├── requirements.txt           pinned versions used for results/ (environment.yml for conda)
├── scripts/                   one shell script per table or figure (Section 3)
├── check_data.py              verify the SHA256SUMS files and the data
├── train.py                   train + evaluate: proposed decoder (default), a comparison method, an ablation, an electrode subset
├── electrode_importance.py    electrode permutation importance → results/Fig_S8_4/electrode_ranking.txt
├── electrode_failure.py       single-electrode failure at test time (reads the ranking file)
├── inference_cost.py          model size and batch-1 latency of the decoder on this machine
├── make_tables.py             results/raw → tables printed in the terminal + csv files (no training)
├── hingesense/                the library
│   ├── config.py              constants, dataset catalogue, paths
│   ├── dataset.py             parquet reader, segment split, causal normalisation, channel mask, 480-D features, anchors
│   ├── model.py               DriftNet, embedding head, gated fusion, soft-kNN read-out
│   ├── trainer.py             leave-one-dataset-out training / evaluation (one dataset per process)
│   ├── baselines.py           comparison methods (one method × dataset per process)
│   ├── evaluate.py            electrode failure and permutation importance (one dataset per process)
│   ├── ranking.py             writer / parser of the electrode ranking file
│   ├── tables.py              table aggregation, terminal rendering, csv writing
│   ├── plots.py               Fig. S8.4
│   ├── cost.py                parameter count, MACs, latency measurement
│   └── runner.py              parallel job runner, machine information, provenance files
├── Data/                      the 12 evaluated datasets: NoLoad/Data_01 … Data_06, Assembly/Data_07 … Data_12
├── Supplementary/             the 11 auxiliary datasets: NoLoad/Data_aux_01 … 08, Assembly/Data_aux_09 … 11
└── results/
    ├── raw/                   what the programs write: per-dataset json, npy predictions, weights, logs
    └── Table_II/ Table_S8_1/ Table_III/ Table_S8_2/ Table_IV/ Table_S8_3/ Table_S8_4/ Table_S2_3/ Fig_S8_4/
```

Every folder under `results/` except `raw/` holds a `README.md` (which table it is, which commands produce it, which
input files and keys it reads), `<table>.csv` with the values as printed in the paper, `<table>_unrounded.csv` with the
unrounded values, and `provenance.txt` (command, date, machine, software versions). `Fig_S8_4/` additionally holds the
figure and `electrode_ranking.txt`.

## 3. How to obtain each table and figure

The times below are the wall-clock time of all training and evaluation that a table needs when it is reproduced with this
repository on two NVIDIA H100 GPUs (measured with the shipped results' run; a table whose inputs already exist takes seconds). They are not the inference latency reported in the paper (that is Table S2.3). With
one GPU and `JOBS=1` expect roughly ten times longer.

### Table II and Table S8.1

```bash
JOBS=24 GPUS=0,1 bash scripts/table_II.sh      # about 3 hours
JOBS=24 GPUS=0,1 bash scripts/table_S8_1.sh    # same inputs
```

Inside: `python train.py` (proposed decoder, 12 datasets), `python train.py --method all` (the 13 comparison methods
of the paper, KalmanNet is the slowest), `python make_tables.py --only Table_II` (or `Table_S8_1`). Outputs:
`results/Table_II/`, `results/Table_S8_1/`.

Comparison methods on the command line and their names in the paper: `ridge` is Linear, `es` Exponential Smoothing,
`kalman` Kalman Filter, `rf` Random Forest, `mlp` Frame-wise MLP, `tcn` TCN, `gru` GRU, `transformer` Transformer,
`kalmannet` KalmanNet, `calib_ridge` Calibrated Ridge, `calib_kalman` Calib. Ridge + Kalman, `calib_es` Calib. Ridge + ES
and `rawknn` Soft-kNN (w/o learning). Three further variants, `gru_aug`, `tcn_aug` and `rf_aug`, can be run by name
(`python train.py --method gru_aug`); they are not part of the paper's tables and appear only with
`python make_tables.py --only Table_II Table_S8_1 --augmented`.

### Table III and Table S8.2

```bash
JOBS=24 GPUS=0,1 bash scripts/table_III.sh     # about 20 minutes
JOBS=24 GPUS=0,1 bash scripts/table_S8_2.sh
```

Inside: `python train.py`, `python make_tables.py --only Table_III` (or `Table_S8_2`). Outputs: `results/Table_III/`,
`results/Table_S8_2/`.

### Table IV and Table S8.3

```bash
JOBS=24 GPUS=0,1 bash scripts/table_IV.sh      # about 2 hours (3.5 hours when run at the same time as table_II.sh)
JOBS=24 GPUS=0,1 bash scripts/table_S8_3.sh
```

Inside: `python train.py`, then `python train.py --ablation <name>` for `no_drift_correction`,
`no_causal_differential`, `no_channel_mask`, `no_reference_pathway`, `no_noise_perturbation`,
`no_symmetry_augmentation`, `no_augmentation`, then `python make_tables.py --only Table_IV` (or `Table_S8_3`).
Outputs: `results/Table_IV/`, `results/Table_S8_3/`.

### Fig. S8.4

```bash
JOBS=24 GPUS=0,1 bash scripts/fig_S8_4.sh      # about 25 minutes
```

Inside: `python train.py`, `python electrode_importance.py` (writes `results/Fig_S8_4/electrode_ranking.txt` when all
12 datasets are available), `python make_tables.py --only Fig_S8_4`. Output: `results/Fig_S8_4/`.

### Table S8.4

```bash
JOBS=24 GPUS=0,1 bash scripts/table_S8_4.sh    # about 40 minutes
```

Inside: `python train.py`, `python electrode_importance.py`, `python train.py --electrodes top4`,
`python electrode_failure.py`, `python make_tables.py --only Table_S8_4`. The electrodes of the reduced configuration and
of the failure tests are read from the ranking file. Output: `results/Table_S8_4/`.

### Table S2.3

```bash
GPUS=0 bash scripts/table_S2_3.sh              # a few minutes once the decoder of dataset 1 exists
```

Inside: `python train.py --folds 1`, `python inference_cost.py` (one CPU thread and, when a GPU is visible, the GPU),
`python make_tables.py --only Table_S2_3`. Only the rows of the same hardware can be compared with the paper, and
latency depends on the clock and power state of the machine. Output: `results/Table_S2_3/`.

### Options shared by the scripts

All root scripts accept `--folds` (dataset numbers 1–12 as in the paper, e.g. `1,3,7-9`, or names such as `Data_03`;
default all), `--jobs`, `--gpus` (`none` for CPU only), `--rerun` (recompute existing results) and `--dry-run` (print
the commands only). `train.py` additionally takes `--method`, `--ablation`, `--electrodes` (`top4` from the ranking
file, or an explicit list such as `0,1,8,15`) and `--seed` (default 2027; a different seed writes to
`results/raw/decoder_seed<seed>/`). Every (method × dataset) is one process; it writes
`results/raw/<configuration>/<dataset>.json` (metrics), `.npy` (test-segment predictions), `_model.pt` (weights) and
`.log` (command, machine, training trace). A GPU is not required by any script, but the neural comparison methods and
KalmanNet are slow on a CPU; on a CPU-only machine start with `bash scripts/table_III.sh`.

## 4. Data

`Data/NoLoad/` holds `Data_01` … `Data_06` and `Data/Assembly/` holds `Data_07` … `Data_12`; the file number is the
dataset number of the paper. `Supplementary/NoLoad/` holds `Data_aux_01` … `Data_aux_08` and `Supplementary/Assembly/`
holds `Data_aux_09` … `Data_aux_11`. All 23 files are Apache Parquet with the same columns: `sample_id` (frame index),
`xdat_segment` (scan-cycle index), `xdat_command` (commanded drive position, not used by the code), `angle_deg` (hinge
angle from the vision system, the ground truth) and `xdat_000` … `xdat_119` (two-terminal resistance in Ω of the 120
electrode pairs). Channel `k` is the pair `(i, j)` with `i < j` in the order `for i in range(16): for j in range(i+1, 16)`,
i.e. `k = Σ_{a<i}(15−a) + (j−i−1)` (`config.channel_index`); `xdat_000` is E0–E1 and `xdat_119` is E14–E15.

`Data/SHA256SUMS` and `Supplementary/SHA256SUMS` list the SHA-256 digests of the files (`cd Data && sha256sum -c
SHA256SUMS`, likewise for `Supplementary`; `python check_data.py` checks both). The recordings are released under
CC BY 4.0 (`Data/LICENSE`, `Supplementary/LICENSE`); please cite the paper when using them.

## 5. Conventions

Electrodes are E0–E15 (zero-based) in the code and in the supplementary material; one-based electrode-pair names of the
main text such as "4–13" denote E3–E12. Tables are rounded half-up from the unrounded values kept in the
`_unrounded.csv` files; means, worst values and sub-degree counts are computed from the unrounded per-dataset values.

## 6. Reproducibility and hardware

The global seed is 2027 (`config.SEED`); permutation seeds are fixed; `torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` are set by the package.
The shipped `results/` were produced on two NVIDIA H100 80 GB (driver 580.173.02) with Intel Xeon Platinum 8480C,
Python 3.11, PyTorch 2.12.0 + CUDA 13.0, NumPy 2.4.6, pandas 3.0.3, pyarrow 24.0.0, scikit-learn 1.9.0 and
matplotlib 3.11.0; every `provenance.txt`, every training log and every latency json records the CPU, GPU, driver and
library versions of the machine that produced it. A training process needs about 1 GB of GPU memory; the Random Forest
methods use 32 CPU threads per process.

## 7. Citation and license

Please cite the paper (`CITATION.cff`). Code: MIT license. Data: CC BY 4.0.

Authors: Panlong Gu, Hong Zhang and Shaoxing Qu (corresponding author), State Key Laboratory of Fluid Power and
Mechatronic Systems, Zhejiang University, Hangzhou, China.
