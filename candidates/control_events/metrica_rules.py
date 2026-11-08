#!/usr/bin/env python3
"""Which control threshold T do Metrica's own possession rules imply?

The competition's ground truth is Metrica's corrected tracking, and CHALLENGE.md uses Metrica's
event vocabulary. Metrica's public sample games (open data) carry Metrica's own event tagging, so
possession built from those events is the closest legal view of how Metrica counts a change. This
script builds two label sets per sample game and compares their possession rate and spell-length
profile with the DFL flag folded at T (control.fold), which is what ctrl_mean averages over.

  practice : tune_on_sample_data.practice_labels, the team changes on every on-ball event
  strict   : practice, but a team spell made only of BALL LOST / CHALLENGE events (a clearance, a
             header, a touch that is not a pass, carry, shot, recovery or restart) is handed back
             to the previous team, following CHALLENGE.md's "touch without control" and "aerial
             duel won without subsequent control"

Reads only open data: Metrica sample games, the DFL flags in $MTAH_WORK/idsse/npz and the DFL
event labels in $MTAH_WORK/ctrl_events. Never the competition's labels.

    python candidates/control_events/metrica_rules.py
"""

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
ROOT = os.path.dirname(CAND)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, CAND)
import possession_metrics as pm   # noqa: E402
import tune_on_sample_data as tsd  # noqa: E402
import control                     # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(ROOT, "external", "work")
SAMPLE = os.path.join(WORK, "sample", "sample-data-master", "data")
FPS = 25
PER90 = 90 * 60 * FPS
TS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]
CONTROLLED = {"PASS", "SHOT", "CARRY", "RECOVERY", "SET PIECE"}


def strict_labels(game, codes, blocks):
    """practice labels with event-free-of-control team spells handed back to the previous team."""
    runs = [(a, b + 1) for _, a, b in blocks]
    ctl = np.zeros((2, len(codes)), bool)
    for e in game.events:
        if e["team"] is not None and e["type"] in CONTROLLED and 0 <= e["start"] < len(codes):
            ctl[e["team"], e["start"]] = True
    out = codes.copy()
    for a, b, c in pm.segments(codes, runs):
        if c not in (pm.H, pm.A) or a == 0 or out[a - 1] not in (pm.H, pm.A) or out[a - 1] == c:
            continue
        if not ctl[0 if c == pm.H else 1, a:b].any():
            out[a:b] = out[a - 1]
    return out


def profile(codes, runs):
    sp = pm.possession_spells(codes, runs)
    dur = np.array([(b - a) / FPS for a, b, _ in sp])
    graded = sum(b - a for a, b in runs)
    return dict(per90=len(sp) / graded * PER90, n=len(sp), dur=dur,
                lt1=float((dur < 1).mean()), lt2=float((dur < 2).mean()),
                lt3=float((dur < 3).mean()), med=float(np.median(dur)))


def ks(a, b):
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def metrica_games():
    out = []
    for folder in sorted(glob.glob(os.path.join(SAMPLE, "Sample_Game_*"))):
        g = (tsd.load_epts_game(folder) if glob.glob(os.path.join(folder, "*.json"))
             else tsd.load_csv_game(folder))
        codes, blocks = tsd.practice_labels(g)
        out.append((g.name, g, codes, blocks))
    return out


def dfl_games():
    out = []
    for path in sorted(glob.glob(os.path.join(WORK, "idsse", "npz", "*.npz"))):
        match = os.path.basename(path)[:-4]
        z = np.load(path)
        status, poss = z["status"], z["poss"]
        gt = np.full(len(status), pm.X, np.int8)
        gt[(status == 1) & (poss == 1)] = pm.H
        gt[(status == 1) & (poss == 2)] = pm.A
        gt[status == 0] = pm.D
        ev = os.path.join(WORK, "ctrl_events", match + ".npy")
        out.append((match, gt, np.load(ev) if os.path.isfile(ev) else None))
    return out


def pooled(profiles):
    n = sum(p["n"] for p in profiles)
    per90 = np.mean([p["per90"] for p in profiles])
    dur = np.concatenate([p["dur"] for p in profiles])
    return dict(per90=per90, n=n, dur=dur, lt1=float((dur < 1).mean()),
                lt2=float((dur < 2).mean()), lt3=float((dur < 3).mean()),
                med=float(np.median(dur)))


def row(name, p):
    return "{:<34} {:>7.0f} {:>6} {:>7.2f} {:>6.3f} {:>6.3f} {:>6.3f}".format(
        name, p["per90"], p["n"], p["med"], p["lt1"], p["lt2"], p["lt3"])


def main():
    head = "{:<34} {:>7} {:>6} {:>7} {:>6} {:>6} {:>6}".format(
        "labels", "poss/90", "spells", "med s", "<1s", "<2s", "<3s")
    print("Possession rate per 90 min of graded time, and spell-length profile\n")
    print(head)
    met = {"practice": [], "strict": []}
    for name, g, codes, blocks in metrica_games():
        runs = [(a, b + 1) for _, a, b in blocks]
        strict = strict_labels(g, codes, blocks)
        for k, c in (("practice", codes), ("strict", strict)):
            p = profile(c, runs)
            met[k].append(p)
            print(row("Metrica {} {}".format(name[-6:], k), p))
    for k in met:
        met[k] = pooled(met[k])
        print(row("Metrica pooled " + k, met[k]))
    print()
    dfl = {T: [] for T in TS}
    evp = []
    for match, gt, ev in dfl_games():
        runs = pm.valid_runs(gt)
        for T in TS:
            dfl[T].append(profile(control.fold(gt, T, runs), runs))
        if ev is not None:
            evp.append(profile(ev, pm.valid_runs(ev)))
    for T in TS:
        dfl[T] = pooled(dfl[T])
        print(row("DFL flag folded T={:g}s".format(T), dfl[T]))
    if evp:
        evp = pooled(evp)
        print(row("DFL event labels", evp))

    print("\nImplied T: the DFL fold whose spell-length distribution is closest (KS) to each\n"
          "label set, and the T whose possession rate matches (linear interpolation).\n")
    rates = np.array([dfl[T]["per90"] for T in TS])
    targets = [("Metrica practice", met["practice"]), ("Metrica strict", met["strict"])]
    if evp:
        targets.append(("DFL event labels", evp))
    for name, p in targets:
        k = [ks(p["dur"], dfl[T]["dur"]) for T in TS]
        r = p["per90"]
        t_rate = float(np.interp(-r, -rates, TS))   # rates fall as T grows
        print("{:<18} KS by T: {}  -> best T={:g}s; rate {:.0f}/90 -> T~{:.2f}s".format(
            name, " ".join("{:g}:{:.3f}".format(T, v) for T, v in zip(TS, k)),
            TS[int(np.argmin(k))], r, t_rate))


if __name__ == "__main__":
    main()
