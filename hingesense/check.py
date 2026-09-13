# -*- coding: utf-8 -*-
"""数据校验：按 Data/ 与 Supplementary/ 的 SHA256SUMS 校验全部 parquet；自检读取、切分、全局通道掩膜与特征构造，不打印细节。
由根目录 check_data.py 调用；其他根脚本启动时只做校验和检查。

Data check: verify every parquet file against the SHA256SUMS of Data/ and Supplementary/; self-check of reading, splitting,
the global channel mask and feature construction. Called by check_data.py; the other root scripts only verify the checksums.
"""
import hashlib
import os

from . import config as C
from . import dataset as ds


def verify_checksums(data_dir=C.DATA_DIR):
    """按 Data/SHA256SUMS 与 Supplementary/SHA256SUMS 校验全部 parquet；返回校验的文件数，任何不符或缺失即抛出异常。
    Verify every parquet file against both SHA256SUMS files; returns the number of files, raises on any mismatch or missing file."""
    return _verify_manifest(data_dir) + _verify_manifest(ds.supplementary_dir(data_dir))


def _verify_manifest(data_dir):
    manifest = os.path.join(data_dir, "SHA256SUMS")
    if not os.path.exists(manifest):
        raise FileNotFoundError("%s not found" % manifest)
    n = 0
    for line in open(manifest, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(None, 1)
        p = os.path.join(data_dir, name.lstrip("*").strip())
        if not os.path.exists(p):
            raise FileNotFoundError("data file missing: %s" % p)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            raise ValueError("checksum mismatch: %s" % p)
        n += 1
    return n


def self_check(data_dir=C.DATA_DIR):
    """段边界与形状检查；推导全局通道掩膜并试构造一个数据集的特征。返回掩膜，不打印。 / Segment and shape checks; derive the global channel mask and build the features of one dataset. Returns the mask, prints nothing."""
    names = C.FOLDS + C.HOLDOUT
    for n in names:
        z = ds.load_raw(n, data_dir)
        if not (z["s1"] < z["q0"] <= z["t0"]):
            raise ValueError("segment order violated in %s" % n)
        if z["x_clean"].shape[1] != C.N_CHANNEL:
            raise ValueError("%s: expected %d channels" % (n, C.N_CHANNEL))
    mask = ds.compute_channel_mask(names, data_dir)
    fd = ds.load_fold(C.FOLDS[0], mask, data_dir)
    if fd["x"].shape != (len(fd["y"]), ds.feature_dim()) or len(ds.support_indices(fd)) == 0:
        raise ValueError("feature construction failed for %s" % C.FOLDS[0])
    return mask
