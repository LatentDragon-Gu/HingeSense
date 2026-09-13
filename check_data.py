#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data check: verify the 23 parquet files against the SHA256SUMS of Data/ and Supplementary/, check the segment split of every dataset,
derive the global channel mask and build the features of one dataset. Prints two lines.
    python check_data.py

数据校验与自检：按 Data/ 与 Supplementary/ 的 SHA256SUMS 校验 23 个 parquet，检查每个数据集的三段划分，
推导全局通道掩膜并试构造一个数据集的特征。只打印两行。"""
import argparse
import time

from hingesense import config as C
from hingesense import check


def main():
    ap = argparse.ArgumentParser(description="verify the data files and run the data self-check")
    ap.add_argument("--data-dir", default=C.DATA_DIR)
    ap.add_argument("--skip-checksum", action="store_true", help="do not verify the SHA256SUMS files")
    args = ap.parse_args()
    t0 = time.time()
    if not args.skip_checksum:
        print("checksums OK (%d files)" % check.verify_checksums(args.data_dir))
    check.self_check(args.data_dir)
    print("CHECK_OK  (%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
