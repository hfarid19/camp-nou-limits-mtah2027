#!/usr/bin/env python3
"""Tables for README.md from grid.pkl / real.pkl (sim.py). Prints markdown.

    python candidates/touch_ceiling/report.py [--metric raw|fine3]

Score of a (config, setting) = mean over the 48 games of the seed-averaged spellF1. "Beats v6" =
mean paired difference vs v6's out-of-fold codes > 0 (SE over games printed where it matters).
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

MAIN = ("A", "B", "C1_25", "C1s_12", "C2", "C3r", "C3", "C13", "C123")


def load(name):
    z = pickle.load(open(os.path.join(C.OUT, name + ".pkl"), "rb"))
    z["idx"] = {c: i for i, c in enumerate(z["cfgs"])}
    z["sidx"] = {s: i for i, s in enumerate(z["settings"])}
    return z


def per_game(z, cfg, st, metric):
    x = z["S"][:, z["idx"][cfg], :, z["sidx"][st], :].mean(1)          # (games, 3)
    return x[:, 0] if metric == "raw" else x.mean(1)


def base(z, metric):
    return z["V"][:, 0] if metric == "raw" else z["V"].mean(1)


def diff(z, cfg, st, metric):
    d = per_game(z, cfg, st, metric) - base(z, metric)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum())


def fmt_cfg(c):
    r, rc, p, a, s = c
    return "r {:g}, r_chg {}, p {:g}, a {:g}, σ {:g}".format(r, "=r" if rc is None else "{:g}".format(rc), p, a, s)


def min_r(z, st, metric, rc, p, a, s, rs):
    """Smallest r in the grid such that every r' >= r beats v6 (None if r = 1 does not)."""
    ok = None
    for r in sorted(rs, reverse=True):
        cfg = (r, rc, p, a, s)
        if cfg not in z["idx"]:
            continue
        if diff(z, cfg, st, metric)[0] > 0:
            ok = r
        else:
            break
    return ok


def grid_tables(z, metric, settings):
    import sim
    out = []
    for rc in sim.GRID_RCHG:
        out.append("\n**r_chg = {}** -- cell = smallest r (grid {}) that beats v6 at that p; '-' = not even "
                   "r = 1\n".format("r" if rc is None else rc, ", ".join("{:g}".format(r) for r in sim.GRID_R)))
        for st in settings:
            out.append("\n{}\n".format(st))
            out.append("| a, σ | " + " | ".join("p {:g}".format(p) for p in sim.GRID_P) + " |")
            out.append("|---" * (len(sim.GRID_P) + 1) + "|")
            for a in sim.GRID_A:
                for s in sim.GRID_S:
                    cells = []
                    for p in sim.GRID_P:
                        rs = [r for r in sim.GRID_R if rc is None or rc < r]
                        m = min_r(z, st, metric, rc, p, a, s, rs)
                        cells.append("-" if m is None else "{:g}".format(m))
                    out.append("| {:g}, {:g} | ".format(a, s) + " | ".join(cells) + " |")
    return "\n".join(out)


HONEST = 0.729   # v6's honest transfer scores: DFL LOGO fine4 0.7286, Metrica holdout 0.7258 / dev 0.7285


def beats(z, cfg, st, metric, bar=None):
    if bar is None:
        return diff(z, cfg, st, metric)[0] > 0
    return per_game(z, cfg, st, metric).mean() > bar


def diagonal(z, metric, settings, bar=None):
    """Uniform quality q = r = p (r_chg = r): smallest q that beats v6 (or reaches `bar`), per (a, sigma)."""
    import sim
    out = ["| setting | " + " | ".join("a {:g} σ {:g}".format(a, s) for a in sim.GRID_A for s in sim.GRID_S) + " |",
           "|---" * (len(sim.GRID_A) * len(sim.GRID_S) + 1) + "|"]
    for st in settings:
        cells = []
        for a in sim.GRID_A:
            for s in sim.GRID_S:
                ok = None
                for q in sorted(sim.GRID_R, reverse=True):
                    if beats(z, (q, None, q, a, s), st, metric, bar):
                        ok = q
                    else:
                        break
                cells.append("-" if ok is None else "{:g}".format(ok))
        out.append("| {} | ".format(st) + " | ".join(cells) + " |")
    return "\n".join(out)


def score_table(z, cfgs, settings, metric, labels=None):
    out = ["| config | " + " | ".join(settings) + " |", "|---" * (len(settings) + 1) + "|"]
    for i, c in enumerate(cfgs):
        cells = []
        for st in settings:
            m = per_game(z, c, st, metric).mean()
            d, se, w = diff(z, c, st, metric)
            cells.append("{:.3f} ({:+.3f})".format(m, d))
        out.append("| {} | ".format(labels[i] if labels else fmt_cfg(c)) + " | ".join(cells) + " |")
    return "\n".join(out)


def one_knob(z, metric, settings):
    """From perfect, degrade one knob; first value (walking down) that no longer beats v6."""
    import sim
    knobs = {"r (r_chg = r)": lambda v: (v, None, 1.0, 1.0, 0), "r_chg only": lambda v: (1.0, v, 1.0, 1.0, 0),
             "p": lambda v: (1.0, None, v, 1.0, 0), "a": lambda v: (1.0, None, 1.0, v, 0)}
    vals = (0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.43, 0.3)
    out = ["| setting | perfect | " + " | ".join("{} break-even".format(k) for k in knobs) + " | σ break-even |",
           "|---" * (len(knobs) + 3) + "|"]
    for st in settings:
        perf = per_game(z, (1.0, None, 1.0, 1.0, 0), st, metric).mean()
        cells = []
        for k, f in knobs.items():
            last = None
            if diff(z, (1.0, None, 1.0, 1.0, 0), st, metric)[0] > 0:
                last = 1.0
                for v in vals:
                    if diff(z, f(v), st, metric)[0] > 0:
                        last = v
                    else:
                        break
            cells.append("-" if last is None else "≥ {:g}".format(last) if last > 0.3 else "< 0.3 ok")
        last = None
        if diff(z, (1.0, None, 1.0, 1.0, 0), st, metric)[0] > 0:
            last = 0
            for s in (1, 2, 3, 4, 5, 6, 8, 10, 15):
                if diff(z, (1.0, None, 1.0, 1.0, s), st, metric)[0] > 0:
                    last = s
                else:
                    break
        cells.append("-" if last is None else "≤ {:g}".format(last))
        out.append("| {} | {:.3f} | ".format(st, perf) + " | ".join(cells) + " |")
    return "\n".join(out)


def sensitivity(z, metric, settings):
    """Score drop from perfect for a fixed step on each knob, and gain at the real point from fixing one knob."""
    import sim
    R = sim.REAL
    steps = [("r 1->0.9", (0.9, None, 1.0, 1.0, 0)), ("r_chg 1->0.7", (1.0, 0.7, 1.0, 1.0, 0)),
             ("p 1->0.9", (1.0, None, 0.9, 1.0, 0)), ("a 1->0.95", (1.0, None, 1.0, 0.95, 0)),
             ("σ 0->4", (1.0, None, 1.0, 1.0, 4))]
    fixes = [("r,r_chg -> 1", (1.0, None) + R[2:]), ("r -> 1 (r_chg 0.2)", (1.0,) + R[1:]),
             ("r_chg -> r", (R[0], None) + R[2:]), ("p -> 1", R[:2] + (1.0,) + R[3:]),
             ("a -> 1", R[:3] + (1.0, R[4])), ("σ -> 0", R[:4] + (0,))]
    perf = (1.0, None, 1.0, 1.0, 0)
    out = ["| setting | " + " | ".join(k for k, _ in steps) + " || at real point | " + " | ".join(k for k, _ in fixes) + " |",
           "|---" * (len(steps) + len(fixes) + 3) + "|"]
    for st in settings:
        p0 = per_game(z, perf, st, metric).mean()
        r0 = per_game(z, R, st, metric).mean()
        out.append("| {} | ".format(st) + " | ".join("{:+.3f}".format(per_game(z, c, st, metric).mean() - p0) for _, c in steps)
                   + " || {:.3f} | ".format(r0)
                   + " | ".join("{:+.3f}".format(per_game(z, c, st, metric).mean() - r0) for _, c in fixes) + " |")
    return "\n".join(out)


def rchg_table(z, metric, st, p, a, s):
    rs = (0.43, 0.7, 0.9, 1.0)
    rcs = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
    out = ["| r \\ r_chg | " + " | ".join("{:g}".format(x) for x in rcs) + " |", "|---" * (len(rcs) + 1) + "|"]
    for r in rs:
        out.append("| {:g} | ".format(r) + " | ".join("{:+.3f}".format(diff(z, (r, rc, p, a, s), st, metric)[0])
                                                     for rc in rcs) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="raw", choices=["raw", "fine3"])
    ap.add_argument("--settings", default=",".join(MAIN))
    ap.add_argument("--full", action="store_true", help="every setting's grid table")
    a = ap.parse_args()
    import sim
    sts = a.settings.split(",")
    m = a.metric
    print("# metric: {}\n".format(m))
    if os.path.isfile(os.path.join(C.OUT, "real.pkl")):
        z = load("real")
        print("v6 (out-of-fold, shipped rule): raw {:.4f}, fine3 {:.4f} (48 games)\n".format(z["V"][:, 0].mean(),
                                                                                         z["V"].mean(1).mean()))
        print("## perfect touches and the real detector point (score, diff vs v6)\n")
        print(score_table(z, [(1.0, None, 1.0, 1.0, 0), sim.REAL], sim.SETTINGS, m, ["perfect", "real detector"]))
        R = sim.REAL
        print("\nreal point paired vs v6: " + "; ".join("{} {:+.3f}±{:.3f} ({}/48)".format(st, *diff(z, R, st, m))
                                                         for st in sim.SETTINGS))
        print("\n## one knob at a time from perfect (others perfect): worst value still beating v6\n")
        print(one_knob(z, m, sim.SETTINGS))
        print("\n## sensitivity\n")
        print(sensitivity(z, m, sts))
        print("\n## near the real point (score, diff vs v6)\n")
        near = [c for c in z["cfgs"] if c[0] in (0.3, 0.43, 0.6) and c[2] in (0.3, 0.475, 0.6)
                and c[3] in (0.65, 0.75, 0.85) and c[4] == 4 and c[1] == (0.2 if c[0] > 0.2 else None)]
        print(score_table(z, near, ("B", "C1s_12", "C2", "C3r", "C13", "C123"), m))
        for st in ("C2", "C123", "B"):
            for p, aa in ((1.0, 1.0), (0.8, 0.95), (0.475, 0.75)):
                print("\n### r_chg x r, {} (p {:g}, a {:g}, σ 4): diff vs v6\n".format(st, p, aa))
                print(rchg_table(z, m, st, p, aa, 4))
    if os.path.isfile(os.path.join(C.OUT, "grid.pkl")):
        z = load("grid")
        print("\n## uniform quality r = p = q (r_chg = r): smallest q beating v6\n")
        print(diagonal(z, m, sim.SETTINGS))
        print("\n## uniform quality, honest bar: smallest q with mean score > {} (A, B only: C* inherit v6's inflated PFF codes)\n".format(HONEST))
        print(diagonal(z, m, ("A", "B"), HONEST))
        print("\n## grid: smallest r beating v6\n")
        print(grid_tables(z, m, sim.SETTINGS if a.full else ("A", "B", "C2", "C123")))


if __name__ == "__main__":
    main()
