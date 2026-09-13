# HingeSense

[English version: README.md](README.md)

下文文章的代码与数据

> P. Gu, H. Zhang, S. Qu, *A Geometry-Preserving Self-Sensing SCM Hinge with Calibration-Anchored Decoding for
> Sub-Degree Joint Proprioception*, IEEE Transactions on Industrial Electronics.

本仓库产出正文的 Table II、III、IV，补充材料的 Table S8.1、S8.2、S8.3、S8.4 与 Fig. S8.4，以及在运行机器上测得的 Table S2.3。每一项对应 `scripts/` 里的一个 shell 脚本，脚本在终端打印该表，并把它以 csv 写入 `results/`。`results/` 已放有一次完整运行的产物，包括训练好的权重与测试段预测，因此不训练也能重建各表。

## 1. 快速开始

```bash
conda create -n hingesense python=3.11 && conda activate hingesense
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130   # 按你的 CUDA 版本选择索引
pip install -r requirements.txt
JOBS=24 GPUS=0,1 bash scripts/table_IV.sh                                        # 训练并打印 Table IV
```

每个脚本先校验数据文件（约十秒），再依次运行该表需要的训练或评估命令，最后打印表格并写入 `results/<表>/`。`results/raw/` 下已有的结果会被复用，所以脚本可以中断后重新运行；共用输入的脚本（所有脚本都共用解码器训练）不会重复计算。只用仓库自带的原始结果重建全部表格、不做任何训练：

```bash
python make_tables.py
```

脚本由三个环境变量控制：`JOBS`（并行进程数，默认 1）、`GPUS`（轮流使用的 GPU 编号，如 `0,1`；默认全部可见 GPU）、`PYTHON`（解释器，默认 `python`）。Windows 下没有 bash 时，按第 3 节列出的 python 命令逐条运行即可。

## 2. 目录结构

```
HingeSense/
├── README.md, README_cn.md    英文说明与本文件
├── LICENSE                    MIT（代码）；Data/LICENSE 与 Supplementary/LICENSE 为 CC BY 4.0（数据）
├── CITATION.cff
├── requirements.txt           产生 results/ 时使用的版本（conda 见 environment.yml）
├── scripts/                   每张表或图一个 shell 脚本（第 3 节）
├── check_data.py              校验 SHA256SUMS 文件与数据
├── train.py                   训练 + 评估：本文解码器（默认）、对比方法、消融、电极子集
├── electrode_importance.py    电极置换重要性 → results/Fig_S8_4/electrode_ranking.txt
├── electrode_failure.py       测试期单电极失效（读取排名文件）
├── inference_cost.py          本机上解码器的模型体量与批 1 时延
├── make_tables.py             results/raw → 终端打印的表 + csv 文件（不训练）
├── hingesense/                程序库
│   ├── config.py              常量、数据集编目、路径
│   ├── dataset.py             parquet 读取、三段切分、因果归一化、通道掩膜、480 维特征、锚
│   ├── model.py               DriftNet、嵌入头、门控融合、soft-kNN 读出
│   ├── trainer.py             留一数据集训练 / 评估（每个数据集一个进程）
│   ├── baselines.py           对比方法（每个方法 × 数据集一个进程）
│   ├── evaluate.py            单电极失效与置换重要性（每个数据集一个进程）
│   ├── ranking.py             电极排名文件的写入 / 解析
│   ├── tables.py              表格汇总、终端渲染、csv 写入
│   ├── plots.py               Fig. S8.4
│   ├── cost.py                参数量、MAC、时延测量
│   └── runner.py              并行作业调度、机器信息、provenance 文件
├── Data/                      12 个被测数据集：NoLoad/Data_01 … Data_06，Assembly/Data_07 … Data_12
├── Supplementary/             11 个辅助数据集：NoLoad/Data_aux_01 … 08，Assembly/Data_aux_09 … 11
└── results/
    ├── raw/                   各程序的直接产物：逐数据集 json、npy 预测、权重、日志
    └── Table_II/ Table_S8_1/ Table_III/ Table_S8_2/ Table_IV/ Table_S8_3/ Table_S8_4/ Table_S2_3/ Fig_S8_4/
```

`results/` 下除 `raw/` 外的每个文件夹都有 `README.md`（这是哪张表、由哪些命令产生、读取哪些输入文件与键）、`<表>.csv`（与文章印刷值相同的取整值）、`<表>_unrounded.csv`（未取整值）和 `provenance.txt`（命令、日期、机器、软件版本）。`Fig_S8_4/` 另有图文件与 `electrode_ranking.txt`。

## 3. 每张表、每张图的获得方法

下列时间是用本仓库在两块 NVIDIA H100 上复现该表时，全部训练与评估所需的总耗时（按产生仓库自带结果的那次运行实测；输入已存在的表只需几秒），与文章报告的推理时延（Table S2.3）无关。单 GPU 且 `JOBS=1` 时约为十倍。

### Table II 与 Table S8.1

```bash
JOBS=24 GPUS=0,1 bash scripts/table_II.sh      # 约 3 小时
JOBS=24 GPUS=0,1 bash scripts/table_S8_1.sh    # 输入相同
```

内部依次运行：`python train.py`（本文解码器，12 个数据集）、`python train.py --method all`（文章的 13 种对比方法，KalmanNet 最慢）、`python make_tables.py --only Table_II`（或 `Table_S8_1`）。输出：`results/Table_II/`、`results/Table_S8_1/`。

命令行中的对比方法名与文章中的名称对应如下：`ridge` 为 Linear，`es` 为 Exponential Smoothing，`kalman` 为 Kalman Filter，`rf` 为 Random Forest，`mlp` 为 Frame-wise MLP，`tcn` 为 TCN，`gru` 为 GRU，`transformer` 为 Transformer，`kalmannet` 为 KalmanNet，`calib_ridge` 为 Calibrated Ridge，`calib_kalman` 为 Calib. Ridge + Kalman，`calib_es` 为 Calib. Ridge + ES，`rawknn` 为 Soft-kNN (w/o learning)。另有三种变体 `gru_aug`、`tcn_aug`、`rf_aug`，可按名运行（`python train.py --method gru_aug`）；它们不属于文章表格，只在 `python make_tables.py --only Table_II Table_S8_1 --augmented` 时出现。

### Table III 与 Table S8.2

```bash
JOBS=24 GPUS=0,1 bash scripts/table_III.sh     # 约 20 分钟
JOBS=24 GPUS=0,1 bash scripts/table_S8_2.sh
```

内部依次运行：`python train.py`、`python make_tables.py --only Table_III`（或 `Table_S8_2`）。输出：`results/Table_III/`、`results/Table_S8_2/`。

### Table IV 与 Table S8.3

```bash
JOBS=24 GPUS=0,1 bash scripts/table_IV.sh      # 约 2 小时（与 table_II.sh 同时运行时约 3.5 小时）
JOBS=24 GPUS=0,1 bash scripts/table_S8_3.sh
```

内部依次运行：`python train.py`，然后对 `no_drift_correction`、`no_causal_differential`、`no_channel_mask`、`no_reference_pathway`、`no_noise_perturbation`、`no_symmetry_augmentation`、`no_augmentation` 各运行一次 `python train.py --ablation <名称>`，最后 `python make_tables.py --only Table_IV`（或 `Table_S8_3`）。输出：`results/Table_IV/`、`results/Table_S8_3/`。

### Fig. S8.4

```bash
JOBS=24 GPUS=0,1 bash scripts/fig_S8_4.sh      # 约 25 分钟
```

内部依次运行：`python train.py`、`python electrode_importance.py`（12 个数据集齐全后写出 `results/Fig_S8_4/electrode_ranking.txt`）、`python make_tables.py --only Fig_S8_4`。输出：`results/Fig_S8_4/`。

### Table S8.4

```bash
JOBS=24 GPUS=0,1 bash scripts/table_S8_4.sh    # 约 40 分钟
```

内部依次运行：`python train.py`、`python electrode_importance.py`、`python train.py --electrodes top4`、`python electrode_failure.py`、`python make_tables.py --only Table_S8_4`。精简配置与失效测试所用的电极均读自排名文件。输出：`results/Table_S8_4/`。

### Table S2.3

```bash
GPUS=0 bash scripts/table_S2_3.sh              # 数据集 1 的解码器存在后只需几分钟
```

内部依次运行：`python train.py --folds 1`、`python inference_cost.py`（单个 CPU 线程，以及有可见 GPU 时的 GPU）、`python make_tables.py --only Table_S2_3`。只有与你硬件相同的行可与文章对照，且时延随机器的时钟与功耗状态变化。输出：`results/Table_S2_3/`。

### 各脚本共用的选项

根目录脚本都接受 `--folds`（文章编号 1–12，如 `1,3,7-9`，或 `Data_03` 这样的名称；默认全部）、`--jobs`、`--gpus`（`none` 表示只用 CPU）、`--rerun`（重算已有结果）和 `--dry-run`（只打印命令）。`train.py` 另有 `--method`、`--ablation`、`--electrodes`（`top4` 取自排名文件，或 `0,1,8,15` 这样的显式列表）与 `--seed`（默认 2027；换种子时写入 `results/raw/decoder_seed<种子>/`）。每个（方法 × 数据集）是一个进程，写出 `results/raw/<配置>/<数据集>.json`（指标）、`.npy`（测试段预测）、`_model.pt`（权重）与 `.log`（命令、机器、训练过程）。所有脚本都不强制要求 GPU，但神经网络对比方法与 KalmanNet 在 CPU 上很慢；只有 CPU 的机器建议先跑 `bash scripts/table_III.sh`。

## 4. 数据

`Data/NoLoad/` 为 `Data_01` … `Data_06`，`Data/Assembly/` 为 `Data_07` … `Data_12`，文件编号即文章的数据集编号。`Supplementary/NoLoad/` 为 `Data_aux_01` … `Data_aux_08`，`Supplementary/Assembly/` 为 `Data_aux_09` … `Data_aux_11`。23 个文件均为列结构相同的 Apache Parquet：`sample_id`（帧序号）、`xdat_segment`（扫描周期编号）、`xdat_command`（指令驱动位置，代码不使用）、`angle_deg`（视觉系统给出的铰链角度，即真值）、`xdat_000` … `xdat_119`（120 个电极对的两端电阻，单位 Ω）。通道 `k` 对应电极对 `(i, j)`（`i < j`），顺序为 `for i in range(16): for j in range(i+1, 16)`，即 `k = Σ_{a<i}(15−a) + (j−i−1)`（`config.channel_index`）；`xdat_000` 为 E0–E1，`xdat_119` 为 E14–E15。

`Data/SHA256SUMS` 与 `Supplementary/SHA256SUMS` 列出各文件的 SHA-256 摘要（`cd Data && sha256sum -c SHA256SUMS`，`Supplementary` 同理；`python check_data.py` 两份都校验）。数据以 CC BY 4.0 发布（`Data/LICENSE`、`Supplementary/LICENSE`），使用时请引用文章。

## 5. 约定

电极在代码与补充材料中为 E0–E15（0 基）；正文中 "4–13" 这样的 1 基电极对指 E3–E12。表格由 `_unrounded.csv` 中的未取整值半进位取整；均值、最差值与亚度计数由未取整的逐数据集值算出。

## 6. 可复现性与硬件

全局种子为 2027（`config.SEED`）；置换种子固定；程序包自行设置 `torch.use_deterministic_algorithms(True)` 与 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。仓库自带的 `results/` 产生于两块 NVIDIA H100 80 GB（驱动 580.173.02）与 Intel Xeon Platinum 8480C，Python 3.11，PyTorch 2.12.0 + CUDA 13.0，NumPy 2.4.6，pandas 3.0.3，pyarrow 24.0.0，scikit-learn 1.9.0，matplotlib 3.11.0；每个 `provenance.txt`、每个训练日志与每个时延 json 都记录了产生它的机器的 CPU、GPU、驱动与库版本。一个训练进程约需 1 GB 显存；Random Forest 方法每个进程使用 32 个 CPU 线程。

## 7. 引用与许可

请引用文章（`CITATION.cff`）。代码：MIT 许可。数据：CC BY 4.0。

作者：Panlong Gu、Hong Zhang、Shaoxing Qu（通讯作者），浙江大学流体动力与机电系统全国重点实验室，杭州。
