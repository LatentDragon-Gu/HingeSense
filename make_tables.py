#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the tables and the figure from the raw results in results/raw: printed in the terminal in the layout of the paper and written to results/<item>/
(<table>.csv rounded as printed, <table>_unrounded.csv unrounded, provenance.txt command and machine). No training; missing inputs are listed with the commands that produce them.
    python make_tables.py                          # all items
    python make_tables.py --only Table_II Fig_S8_4
    python make_tables.py --only Table_II --augmented   # Table II / S8.1 with the three augmented variants (GRU + aug, TCN + aug, Random Forest + aug)
Items: Table_II Table_S8_1 Table_III Table_S8_2 Table_IV Table_S8_3 Table_S8_4 Table_S2_3 Fig_S8_4

把 results/raw 中已有的原始结果整理成各表与图：在终端按文章版式打印，并写入 results/<条目>/
（<表>.csv 文章版式取整值、<表>_unrounded.csv 未取整值、provenance.txt 命令与机器）。不训练；缺少的输入逐条列出并给出生成命令。
--only 选条目；--augmented 让 Table II / S8.1 另加三种增强变体。"""
import argparse
import datetime
import os
import sys

from hingesense import config as C
from hingesense import check
from hingesense import runner
from hingesense import tables as T

def _rel(p):
    p = os.path.abspath(p)
    return os.path.relpath(p, C.ROOT_DIR) if p.startswith(C.ROOT_DIR + os.sep) else p


ITEMS = ["Table_II", "Table_S8_1", "Table_III", "Table_S8_2", "Table_IV", "Table_S8_3", "Table_S8_4", "Table_S2_3", "Fig_S8_4"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--only", nargs="*", default=[], help="items to build; default all")
    ap.add_argument("--augmented", action="store_true", help="add the three augmented comparison methods to Table II / Table S8.1")
    ap.add_argument("--raw", default=C.RAW_DIR, help="directory of the raw results")
    ap.add_argument("--out", default=C.RESULT_DIR, help="directory of the table folders")
    args = ap.parse_args()
    for it in args.only:
        if it not in ITEMS:
            sys.exit("unknown item %r; choose from %s" % (it, " ".join(ITEMS)))
    wanted = [it for it in ITEMS if it in (args.only or ITEMS)]
    check.verify_checksums()
    command = "python " + " ".join(os.path.relpath(a, C.ROOT_DIR) if os.path.isabs(a) else a for a in sys.argv)
    status = []

    def run(fn, items):
        if not set(wanted) & set(items):
            return None
        started = datetime.datetime.now()
        res = fn()
        written = {}
        for tb in res["tables"]:
            if tb.name not in wanted:
                continue
            print("\n" + tb.text() + "\n")
            written.setdefault(tb.name, []).extend(tb.write())
        for name, files in written.items():
            runner.write_provenance(os.path.join(args.out, name, "provenance.txt"), command, started,
                                    inputs=["%d input file(s) missing" % len(res["missing"])] if res["missing"] else ["all inputs present"])
            status.append((name, res, files))
        return res

    run(lambda: T.table_ii_s81(args.raw, args.out, augmented=args.augmented), ["Table_II", "Table_S8_1"])
    run(lambda: T.table_iii_s82(args.raw, args.out), ["Table_III", "Table_S8_2"])
    run(lambda: T.table_iv_s83(args.raw, args.out), ["Table_IV", "Table_S8_3"])
    run(lambda: T.table_s84(args.raw, args.out), ["Table_S8_4"])
    r = run(lambda: T.fig_s84_summary(args.raw, args.out), ["Fig_S8_4"])
    if r is not None and not r["missing"] and os.path.exists(C.RANKING_FILE):
        from hingesense import plots
        files = plots.fig_importance(r["norm_by_fold"], os.path.join(args.out, "Fig_S8_4"))
        status[-1][2].extend(files)
    run(lambda: T.table_s23(args.raw, args.out), ["Table_S2_3"])
    for name, res, files in status:
        if res["missing"]:
            print("%-10s incomplete: %d input file(s) missing; run:  %s" % (name, len(res["missing"]), " ; ".join(res["hints"])))
        else:
            print("%-10s written: %s" % (name, ", ".join(_rel(f) for f in files)))


if __name__ == "__main__":
    main()
