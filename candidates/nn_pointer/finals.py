#!/usr/bin/env python3
"""Round-5 FINAL-RECIPE scoring (arch/round5_prereg.txt addendum 2026-11-10 01:32): MBR decode of fixed member sets,
the decoder config chosen NESTED per DFL game from the pre-specified g1 grid (mbr_decode/exp.grid("g1"), picked on
the other six games with mbr_decode/report.nested_dfl), the full-DFL pick for SkillCorner / Metrica.

    python -I candidates/nn_pointer/finals.py run --ens NAME=m1,m2,... [--ens ...] [--jobs 8]
    python -I candidates/nn_pointer/finals.py report --ens F3 --ens F4 --ref-ens LEADER [--vs F3]

A member is a probs folder name: plain (nn_pff_dfl / skc_mbr / nn_pff_metrica layout, as mbr_decode/data.py) or
"adapt32/<x>" (one folder holding dfl2_* / skc2_* / dev_* files, e.g. the TTA4+a50 TCN). Grid results per unit go to
$MTAH_WORK/arch/r5_arch/mbr/res_g1_<domain>_<ens>.pkl (exp.py's format, "base" = the v6 rule on the hard vote).
report: nested DFL / full-pick SKC & Metrica scores, paired vs v6 (offline v6-rule vote of t5c, m5c, radar_pff) and vs
--ref-ens, poss/90, the SkillCorner matched-possession check (mbr_decode/analysis.matched: the v6 vote re-folded to the
candidate's possession count per game), and for --vs the (DFL + SKC) gain-sum difference with its SE
(sqrt(SE_dfl^2 + SE_skc^2), the two benches being independent).
"""
import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mbr_decode"))
import data      # noqa: E402
import decoder   # noqa: E402
import exp       # noqa: E402
import report    # noqa: E402
import analysis  # noqa: E402
import mbr       # noqa: E402

OUT = os.path.join(data.WORK, "arch", "r5_arch", "mbr")
_orig_path = data._path


def _path(domain, m, u):
    if "/" in m:
        f = {"dfl": "dfl2_{}_{}", "skc": "skc2_{}_2027", "met": "dev_{}_{}", "pff": "pff2_{}_2027"}[domain]
        return os.path.join(data.PROBS, m, f.format(*u) if isinstance(u, tuple) else f.format(u))
    return _orig_path(domain, m, u)


data._path = _path
DOMS = ("dfl", "skc", "met")


def run_unit(args):
    domain, name, members, u = args
    blocks, tg = data.info(domain, u)
    ms = [data.member(domain, m, u) for m in members]
    probs, codes = [p for p, _ in ms], [c for _, c in ms]
    t0 = time.time()
    res = {"base": data.score(domain, decoder.base(codes, blocks), tg)}
    cache = {}
    for cfg in exp.grid("g1"):
        pred, inf = decoder.decode(probs, codes, blocks, cfg, cache)
        r = data.score(domain, pred, tg)
        r["n"] = len(mbr.spells(pred, mbr.runs_of(blocks))[0])
        res[exp.key(cfg)] = r
    return domain, name, u, res, time.time() - t0


def cmd_run(a):
    os.makedirs(OUT, exist_ok=True)
    ens = dict(e.split("=", 1) for e in a.ens)
    tasks = []
    for name, ms in ens.items():
        members = ms.split(",")
        json.dump(members, open(os.path.join(OUT, "members_{}.json".format(name)), "w"))
        for d in DOMS:
            if os.path.isfile(os.path.join(OUT, "res_g1_{}_{}.pkl".format(d, name))):
                continue
            tasks += [(d, name, members, u) for u in data.units(d)]
    out = {}
    with Pool(a.jobs) as pool:
        for d, name, u, res, dt in pool.imap_unordered(run_unit, tasks):
            for k, v in res.items():
                out.setdefault((d, name), {}).setdefault(k, {})[u] = v
            print(d, name, u, "{:.0f}s".format(dt), flush=True)
    for (d, name), R in out.items():
        pickle.dump(R, open(os.path.join(OUT, "res_g1_{}_{}.pkl".format(d, name)), "wb"))


def load(d, name):
    return pickle.load(open(os.path.join(OUT, "res_g1_{}_{}.pkl".format(d, name)), "rb"))


def final_scores(name):
    """-> ({domain: {unit: main}}, {domain: {unit: n spells}}, picks, full)."""
    R = {d: load(d, name) for d in DOMS}
    nd, picks, full = report.nested_dfl(R["dfl"])
    S = {"dfl": nd}
    N = {"dfl": {u: R["dfl"][picks[u[0]]][u]["n"] for u in nd}}
    for d in ("skc", "met"):
        S[d] = {u: r["main"] for u, r in R[d][full].items()}
        N[d] = {u: r["n"] for u, r in R[d][full].items()}
    return S, N, picks, full


def v6_ref():
    out = {}
    for d in DOMS:
        R = report.load("g1", d, "v6")
        out[d] = {u: r["main"] for u, r in R["base"].items()}
    return out


def per90(N, d):
    tot_n, tot_min = 0, 0.0
    for u, n in N.items():
        blocks, _ = data.info(d, u)
        tot_n += n
        tot_min += sum(b - a for a, b in mbr.runs_of(blocks)) / 25.0 / 60.0
    return 90.0 * tot_n / tot_min


def matched_skc(N):
    """v6 vote re-folded to the candidate's possession count per SkillCorner game (analysis.matched)."""
    out = {}
    for u, n in N.items():
        blocks, tg = analysis.targets("skc", u)
        codes = [data.member("skc", m, u)[1] for m in data.V6]
        out[u] = analysis.matched("skc", codes, blocks, n, tg)["main"]
    return out


def cmd_report(a):
    ref6 = v6_ref()
    res = {}
    names = list(a.ens) + ([a.ref_ens] if a.ref_ens and a.ref_ens not in a.ens else [])
    for name in names:
        res[name] = final_scores(name)
    for name in a.ens:
        S, N, picks, full = res[name]
        print("== {} members {}".format(name, json.load(open(os.path.join(OUT, "members_{}.json".format(name))))))
        print("   full-DFL pick {}; nested picks {}".format(dict(full), {g: dict(c) for g, c in picks.items()}))
        for d in DOMS:
            line = "   {:3s} {:.4f} poss/90 {:.1f} | vs v6 {}".format(d, np.mean(list(S[d].values())), per90(N[d], d),
                                                                     report.fmt(data.paired(S[d], ref6[d], d)))
            if a.ref_ens:
                line += " | vs {} {}".format(a.ref_ens, report.fmt(data.paired(S[d], res[a.ref_ens][0][d], d)))
            print(line)
        m6 = matched_skc(N["skc"])
        print("   skc vs v6 at matched possession count {}".format(report.fmt(data.paired(S["skc"], m6, "skc"))))
        if a.vs and a.vs in res and a.vs != name:
            T = res[a.vs][0]
            dd = data.paired(S["dfl"], T["dfl"], "dfl")
            ds = data.paired(S["skc"], T["skc"], "skc")
            dm = np.mean(list(S["met"].values())) - np.mean(list(T["met"].values()))
            print("   vs {}: DFL {} SKC {} | gain-sum {:+.4f} ± {:.4f} | Metrica {:+.4f}".format(
                a.vs, report.fmt(dd), report.fmt(ds), dd[0] + ds[0], float(np.hypot(dd[1], ds[1])), dm))
    if a.ref_ens:
        S, N, picks, full = res[a.ref_ens]
        print("== reference {}: ".format(a.ref_ens) + " ".join("{} {:.4f} (poss/90 {:.1f})".format(
            d, np.mean(list(S[d].values())), per90(N[d], d)) for d in DOMS) +
            " | vs v6 " + " ".join("{} {}".format(d, report.fmt(data.paired(S[d], ref6[d], d))) for d in DOMS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--ens", action="append", default=[])
    ap.add_argument("--ref-ens", default=None)
    ap.add_argument("--vs", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_report(a)


if __name__ == "__main__":
    main()
