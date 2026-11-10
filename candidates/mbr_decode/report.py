#!/usr/bin/env python3
"""Tables from exp.py results: every config (multiplicity), the nested-on-DFL pick, paired vs v6 / lsw.

    python candidates/mbr_decode/report.py --grid g1[+g2] [--ens v6] [--domains dfl,skc,met,pff] [--all]

Nested pick on DFL: for each DFL game, the config with the best mean fine4 over the OTHER 6 games (all
seeds); that game's units are scored with it. The FULL pick (best mean over all 7 DFL games) is the one a
live pipeline would ship; it is applied unchanged to SkillCorner, Metrica and PFF.
References: "base" of the v6 member set (= frozen v6's rule on the offline vote) and of lsw.
"""

import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402

OUT = os.path.join(data.WORK, "arch", "mbr_decode")


def load(grid, domain, ens):
    """grid: one name or several joined by '+' (their configs pooled; base taken from the first)."""
    out = None
    for g in grid.split("+"):
        p = os.path.join(OUT, "res_{}_{}_{}.pkl".format(g, domain, ens))
        if not os.path.isfile(p):
            return None
        R = pickle.load(open(p, "rb"))
        if out is None:
            out = R
        else:
            out.update({k: v for k, v in R.items() if k != "base"})
    return out


def means(R, key):
    return float(np.mean([r["main"] for r in R[key].values()]))


def nested_dfl(R, configs=None):
    cfgs = [k for k in R if k != "base"] if configs is None else configs
    us = sorted(R["base"])
    out, picks = {}, {}
    for g in sorted({u[0] for u in us}):
        tr = [u for u in us if u[0] != g]
        best = max(cfgs, key=lambda c: np.mean([R[c][u]["main"] for u in tr]))
        picks[g] = best
        for u in us:
            if u[0] == g:
                out[u] = R[best][u]["main"]
    full = max(cfgs, key=lambda c: np.mean([R[c][u]["main"] for u in us]))
    return out, picks, full


def fmt(p):
    return "{:+.4f} ± {:.4f} ({}/{})".format(*p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="g1")
    ap.add_argument("--ens", default="v6")
    ap.add_argument("--domains", default="dfl,skc,met,pff")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    doms = a.domains.split(",")
    R = {d: load(a.grid, d, a.ens) for d in doms}
    V6 = {d: load("g1", d, "v6") for d in doms}
    LSW = {d: load("g1", d, "lsw") for d in doms}
    ref6 = {d: {u: r["main"] for u, r in V6[d]["base"].items()} for d in doms if V6[d]}
    refl = {d: {u: r["main"] for u, r in LSW[d]["base"].items()} for d in doms if LSW[d]}
    if a.all:
        print("every config ({} + base), ens {}: mean main score, paired vs v6 base".format(len(R["dfl"]) - 1, a.ens))
        for k in R["dfl"]:
            line = "{:70s}".format(str(k)[:70])
            for d in doms:
                if R[d] and k in R[d]:
                    s = {u: r["main"] for u, r in R[d][k].items()}
                    line += " | {} {:.4f} {}".format(d, np.mean(list(s.values())), fmt(data.paired(s, ref6[d], d)))
            print(line)
    nd, picks, full = nested_dfl(R["dfl"])
    print("\nnested DFL picks:", {g: dict(c) for g, c in picks.items()})
    print("full-DFL pick (shipped config):", dict(full))
    rows = [("v6 (offline vote, v6 rule)", {d: ref6[d] for d in ref6}),
            ("lsw (offline vote, v6 rule)", {d: refl[d] for d in refl}),
            ("MBR {} nested/full-pick".format(a.ens),
             dict({"dfl": nd}, **{d: {u: r["main"] for u, r in R[d][full].items()} for d in doms if d != "dfl" and R[d]}))]
    print("\n{:34s}".format("") + "".join("{:>44s}".format(d) for d in doms))
    for name, S in rows:
        line = "{:34s}".format(name)
        for d in doms:
            if d not in S:
                line += "{:>44s}".format("-")
                continue
            m = np.mean(list(S[d].values()))
            line += "{:>44s}".format("{:.4f} v6 {}".format(m, fmt(data.paired(S[d], ref6[d], d))))
        print(line)
        line = "{:34s}".format("")
        for d in doms:
            if d in S and d in refl:
                line += "{:>44s}".format("vs lsw {}".format(fmt(data.paired(S[d], refl[d], d))))
        print(line)


if __name__ == "__main__":
    main()
