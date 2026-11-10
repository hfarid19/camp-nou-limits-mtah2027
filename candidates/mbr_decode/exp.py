#!/usr/bin/env python3
"""Offline MBR experiments on saved member outputs: every config of the grid on every unit of a bench.

    python candidates/mbr_decode/exp.py --domain dfl --ens v6 --grid g1 [--jobs 4]

Writes $MTAH_WORK/arch/mbr_decode/res_<grid>_<domain>_<ens>.pkl = {config: {unit: score dict}}, with
config "base" = the v6 rule on the hard vote (merge_short_v2 1 s, control.fold 1.25 s; prekick_dead
is a no-op offline), i.e. vote_offline / swap_eval's reference. Every config is logged (multiplicity).
"""

import argparse
import itertools
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data  # noqa: E402
import mbr   # noqa: E402
import decoder  # noqa: E402

OUT = os.path.join(data.WORK, "arch", "mbr_decode")
ENS = {"v6": data.V6, "lsw": data.LSW}


def grid(name):
    """-> list of decoder configs (dicts)."""
    if name == "g1":
        out = []
        for alpha, rho in itertools.product((0.1, 0.3, 1.0), (0.05, 0.2)):
            for margin in (0.0, 0.25, 0.5):
                out.append(dict(src="hmm", alpha=alpha, rho=rho, margin=margin))
        for margin in (0.0, 0.25, 0.5):
            out.append(dict(src="mem", margin=margin))
        return out
    if name == "g2":            # written after seeing g1 on DFL / SKC / Metrica / PFF (logged as such)
        out = []
        for alpha, rho, r, margin in itertools.product((0.3, 1.0), (0.2, 0.5), (2, 4), (0.0, 0.5)):
            if rho == 0.2 and r == 4:
                continue        # = g1 configs
            out.append(dict(src="hmm", alpha=alpha, rho=rho, r=r, margin=margin))
        for margin in (0.0, 0.5):
            out.append(dict(src="mix", alpha=1.0, rho=0.2, margin=margin))
        return out
    if name == "g3":            # exploratory, after the coordinator's "fewer knobs" question: one fold read
        b = dict(src="hmm", alpha=1.0, rho=0.2, margin=0.0)
        return [dict(b, folds=(0.0,)), dict(b, folds=(1.0,))]
    if name.startswith("stab"):  # stability of the picked config: sampling seed and sample count (not a selection grid)
        base = (dict(src="hmm", alpha=1.0, rho=0.2, margin=0.5) if name == "stab1"          # g1 full-DFL pick
                else dict(src="hmm", alpha=0.3, rho=0.5, r=4, margin=0.5))               # g1+g2 full-DFL pick
        return [dict(base, seed=s) for s in (1, 2, 3, 4, 5)] + [dict(base, n=64), dict(base, n=128)]
    raise ValueError(name)


def key(cfg):
    return tuple(sorted(cfg.items()))


def run_unit(args):
    domain, ens, gname, u = args
    blocks, tg = data.info(domain, u)
    ms = [data.member(domain, m, u) for m in ENS[ens]]
    probs = [p for p, _ in ms]
    codes = [c for _, c in ms]
    t0 = time.time()
    res = {"base": data.score(domain, decoder.base(codes, blocks), tg)}
    cache = {}
    for cfg in grid(gname):
        pred, inf = decoder.decode(probs, codes, blocks, cfg, cache)
        r = data.score(domain, pred, tg)
        r.update(n_switched=inf["n_switched"], n_windows=inf["n_windows"], F=inf["F"])
        res[key(cfg)] = r
    return u, res, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--ens", default="v6")
    ap.add_argument("--grid", default="g1")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    us = data.units(a.domain)
    out = {}
    with Pool(a.jobs) as pool:
        for u, res, dt in pool.imap_unordered(run_unit, [(a.domain, a.ens, a.grid, u) for u in us]):
            for k, v in res.items():
                out.setdefault(k, {})[u] = v
            print(u, "{:.1f}s".format(dt), "base {:.4f}".format(res["base"]["main"]), flush=True)
    path = os.path.join(OUT, "res_{}_{}_{}.pkl".format(a.grid, a.domain, a.ens))
    pickle.dump(out, open(path, "wb"))
    print("->", path)


if __name__ == "__main__":
    main()
