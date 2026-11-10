#!/usr/bin/env python3
"""Robustness tables for the g1 pick (coordinator's checks): coarseness (possessions per 90), SkillCorner /
DFL / Metrica gain at a matched possession count, DFL per label reading and sel2.

    python candidates/mbr_decode/analysis.py [--jobs 2]

Decoders compared per unit (lsw = o_ls05, m5c_W, radar_pff; v6 = t5c, m5c, radar_pff):
    v6, lsw   the offline vote with the v6 rule (merge_short_v2 1 s, fold 1.25 s)
    mbr       MBR on the lsw members; DFL: each game decoded with ITS nested pick (chosen on the other 6
              games), other benches: the g1 full-DFL pick (report.py)
Matched count: the v6 (lsw) vote re-folded (merge 1 s, then control.fold at T in 0 ... 4 s, step 0.05) to the
T whose possession count is closest to mbr's on that unit; that unit's score at that T.
Possessions per 90 = spells / graded minutes * 90.
Writes $MTAH_WORK/arch/mbr_decode/analysis.pkl and prints the tables.
"""

import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data     # noqa: E402
import decoder  # noqa: E402
import mbr      # noqa: E402
import report   # noqa: E402

TS = np.round(np.arange(0.0, 4.0001, 0.05), 2)
SEL2 = ("T1", "T1.5", "T2", "evt3", "evt2")
FINE4 = ("raw", "T0.5", "T1", "evt1")


def targets(domain, u):
    if domain == "dfl":
        b, tg = data.vo.dfl_info()[u]
        return b, {k: tg[k] for k in FINE4 + SEL2 if k in tg}
    return data.info(domain, u)


def scores(domain, pred, tg):
    r = {k: data.f1(g, pred) for k, g in tg.items()}
    if domain == "dfl":
        r["main"] = float(np.mean([r[k] for k in FINE4]))
        r["sel2"] = float(np.mean([r[k] for k in SEL2]))
    elif domain == "skc":
        r["main"] = float(np.mean([r[k] for k in ("raw", "T0.5", "T1")]))
    else:
        r["main"] = r["raw"]
    return r


def nsp(codes, runs):
    return len(mbr.spells(codes, runs)[0])


def matched(domain, codes, blocks, target_n, tg):
    runs = mbr.runs_of(blocks)
    best = None
    for T in TS:
        c = decoder.rule(codes, blocks, 1.0, float(T))
        n = nsp(c, runs)
        k = (abs(n - target_n), abs(T - 1.25))
        if best is None or k < best[0]:
            best = (k, T, c, n)
    _, T, c, n = best
    return dict(T=float(T), n=n, **scores(domain, c, tg))


def run_unit(args):
    domain, u, cfg = args
    blocks, tg = targets(domain, u)
    runs = mbr.runs_of(blocks)
    graded = sum(b - a for a, b in runs)
    out = {"min90": graded / 25.0 / 60.0 / 90.0}
    ens = {}
    for name, ms in (("v6", data.V6), ("lsw", data.LSW)):
        mm = [data.member(domain, m, u) for m in ms]
        ens[name] = ([p for p, _ in mm], [c for _, c in mm])
        c = decoder.base(ens[name][1], blocks)
        out[name] = dict(n=nsp(c, runs), dead=float((c[np.concatenate([np.arange(a, b) for a, b in runs])] == 2).mean()),
                         **scores(domain, c, tg))
    c, _ = decoder.decode(ens["lsw"][0], ens["lsw"][1], blocks, cfg)
    out["mbr"] = dict(n=nsp(c, runs), dead=float((c[np.concatenate([np.arange(a, b) for a, b in runs])] == 2).mean()),
                      **scores(domain, c, tg))
    out["frac_changed_vs_lsw"] = float((c != decoder.base(ens["lsw"][1], blocks)).mean())
    for name in ("v6", "lsw"):
        out[name + "_matched"] = matched(domain, ens[name][1], blocks, out["mbr"]["n"], tg)
    return domain, u, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    R = report.load("g1", "dfl", "lsw")
    _, picks, full = report.nested_dfl(R)
    full = dict(full)
    tasks = [("dfl", u, dict(picks[u[0]])) for u in data.units("dfl")]
    tasks += [(d, u, full) for d in ("skc", "met") for u in data.units(d)]
    res = {}
    with Pool(a.jobs) as pool:
        for d, u, out in pool.imap_unordered(run_unit, tasks):
            res.setdefault(d, {})[u] = out
    pickle.dump(res, open(os.path.join(report.OUT, "analysis.pkl"), "wb"))
    show(res)


def show(res):
    for d in ("dfl", "skc", "met"):
        R = res[d]
        us = sorted(R)
        print("\n== {} ({} units)".format(d, len(us)))
        for name in ("v6", "lsw", "mbr"):
            per90 = np.mean([R[u][name]["n"] / R[u]["min90"] for u in us])
            dead = np.mean([R[u][name]["dead"] for u in us])
            print("  {:4s} poss/90 {:6.1f}  dead share {:.3f}  main {:.4f}".format(
                name, per90, dead, np.mean([R[u][name]["main"] for u in us])))
        print("  mbr frames changed vs lsw: {:.2%}".format(np.mean([R[u]["frac_changed_vs_lsw"] for u in us])))
        m = {u: R[u]["mbr"]["main"] for u in us}
        for ref in ("v6", "lsw"):
            r0 = {u: R[u][ref]["main"] for u in us}
            r1 = {u: R[u][ref + "_matched"]["main"] for u in us}
            Ts = [R[u][ref + "_matched"]["T"] for u in us]
            print("  mbr vs {:3s}: as is {}   at matched count {} (refold T median {:.2f}, range {:.2f}-{:.2f})".format(
                ref, report.fmt(data.paired(m, r0, d)), report.fmt(data.paired(m, r1, d)),
                np.median(Ts), min(Ts), max(Ts)))
        if d == "dfl":
            for k in FINE4 + ("main",) + SEL2 + ("sel2",):
                line = "  reading {:5s}".format(k)
                for ref in ("v6", "lsw"):
                    line += "  vs {} {}".format(ref, report.fmt(data.paired(
                        {u: R[u]["mbr"][k] for u in us}, {u: R[u][ref][k] for u in us}, d)))
                print(line)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--show":
        show(pickle.load(open(os.path.join(report.OUT, "analysis.pkl"), "rb")))
    else:
        main()
