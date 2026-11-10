#!/usr/bin/env python3
"""Size x score table of nn_asformer configurations (the scaling study / architecture sweep).

    python candidates/nn_asformer/report.py CONFIG [CONFIG ...] [--ref ensemble_ctrl_v5c] [--games-min 48]

For each CONFIG (a folder under $MTAH_WORK/arch/nn_asformer with fold_pff_<k>.npz and PFF
out-of-fold codes under $MTAH_WORK/probs/nn_asformer/CONFIG): parameters, weight file size, mean
training loss at the exported epoch and mean validation CE (training-side split) over the fold
models, epochs run, and the PFF out-of-fold scores on the selection games present (fine3 / raw,
paired vs the reference codes: mean difference, SE over games, wins). Rows with fewer games than
--games-min are marked partial. Writes nothing.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(CAND, "pff"), CAND]
import bench_pff  # noqa: E402

WORK = bench_pff.WORK
ARCH = os.path.join(WORK, "arch", "nn_asformer")


def sel_games():
    return sorted(g for f in json.load(open(os.path.join(WORK, "arch", "nn_pff", "selection_folds.json")))["folds"] for g in f)


def find_pff(name):
    for root in ("nn_asformer", "nn_pff", "pff"):
        d = os.path.join(WORK, "probs", root, name)
        if os.path.isdir(d):
            return d
    return None


def scores(name, gs):
    d = find_pff(name)
    out = {}
    if d is None:
        return out
    for g in gs:
        f = os.path.join(d, "pff2_{}_2027_codes.npy".format(g))
        if os.path.isfile(f):
            out[g] = bench_pff.score_codes(np.load(f).astype(np.int8), g)
    return out


def model_stats(name):
    files = sorted(glob.glob(os.path.join(ARCH, name, "fold_pff_*.npz")))
    if not files:
        return None
    st = [json.loads(str(np.load(f)["stats"])) for f in files]
    cfg = json.loads(str(np.load(files[0])["cfg"]))
    return dict(params=st[0]["params"], mb=os.path.getsize(files[0]) / 1e6, n=len(files),
                train=np.mean([s["train_loss_at_best"] for s in st]), val=np.mean([s["best_val_ce"] for s in st]),
                epochs=np.mean([s["epochs_run"] for s in st]), best_ep=np.mean([s["best_epoch"] for s in st]),
                minutes=np.mean([s["minutes"] for s in st]), cfg=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--ref", default="ensemble_ctrl_v5c")
    ap.add_argument("--games-min", type=int, default=48)
    a = ap.parse_args()
    gs = sel_games()
    ref = scores(a.ref, gs)
    print("{:14s} {:>9s} {:>6s} {:>4s} {:>7s} {:>7s} {:>5s} {:>3s} | {:>6s} {:>6s} {:>6s}  {:>22s}  {:>22s}".format(
        "config", "params", "MB", "fold", "train", "val_ce", "ep", "n", "fine3", "raw", "poss", "fine3 vs ref", "raw vs ref"))
    for name in a.configs:
        ms = model_stats(name)
        sc = scores(name, gs)
        common = sorted(set(sc) & set(ref))
        line = "{:14s} ".format(name[:14])
        if ms:
            arch = ms["cfg"]
            line += "{:9d} {:6.1f} {:4d} {:7.4f} {:7.4f} {:5.1f} ".format(ms["params"], ms["mb"], ms["n"], ms["train"], ms["val"], ms["best_ep"])
        else:
            line += " " * 44
        line += "{:3d} | ".format(len(sc))
        if common:
            f3 = np.mean([sc[g]["fine3"] for g in common])
            rw = np.mean([sc[g]["raw"] for g in common])
            line += "{:6.4f} {:6.4f} {:6d}".format(f3, rw, sum(sc[g]["n_pred"] for g in common))
            for k in ("fine3", "raw"):
                dd = np.array([sc[g][k] - ref[g][k] for g in common])
                line += "  {:+.4f}±{:.4f} ({:2d}/{:2d})".format(dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else 0, int((dd > 0).sum()), len(dd))
            if len(common) < a.games_min:
                line += "  partial"
        if ms:
            c = ms["cfg"]
            line += "   [{} d{} L{} S{} h{}{}]".format(c["arch"], c["d"], c["L"], c["S"], c["heads"],
                                                  " N{} e{}".format(c["ssm_N"], c["ssm_expand"]) if c["arch"] == "temba" else " w{}".format(c["w_max"]))
        print(line)
    if common:
        line = "{:14s} {:44s}{:3d} | {:6.4f} {:6.4f} {:6d}".format(a.ref[:14], "", len(ref), np.mean([ref[g]["fine3"] for g in ref]),
                                                                    np.mean([ref[g]["raw"] for g in ref]), sum(ref[g]["n_pred"] for g in ref))
        print(line)


if __name__ == "__main__":
    main()
