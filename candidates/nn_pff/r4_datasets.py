#!/usr/bin/env python3
"""Round 4 training sets, assembled by hard links from ds_pff2 + the new files of ds_r4new.

    python -I candidates/nn_pff/r4_datasets.py tgt  [--src $MTAH_WORK/ds_r4new]   # target cache (needs raw/)
    python -I candidates/nn_pff/r4_datasets.py link [--work $MTAH_WORK]            # build the 3 folders

ds_r4new (make_dataset.py, see README "Round 4") holds only new degradations:
  the 16 PFF holdout games x seeds 41 42 43 45 46 47 48 49 (--allow-holdout --holdout-games),
  the 48 PFF selection games x seeds 45-49, the 7 DFL games x seeds 45-49 (all degrade_v2).
Folders built (ds/ and tgt/; no raw/: data.targets reads the tgt cache):
  ds_pff64    = ds_pff2 + holdout games x 41-43           (64 PFF x 3 seeds + 7 DFL x 3; experiment 1)
  ds_pff8     = ds_pff2 + selection x 45-49 + DFL x 45-49 (48 PFF x 8 + 7 DFL x 8; experiment 2)
  ds_pff8_64  = ds_pff8 + holdout games x 41-43, 45-49    (64 PFF x 8 + 7 DFL x 8; experiments 1 + 2)
Training on a folder with holdout games needs NN_PFF_ALLOW_HOLDOUT=1 (data.py).
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402

NEW = (45, 46, 47, 48, 49)
OLD = (41, 42, 43)


def seed_of(f):
    return int(os.path.basename(f)[:-4].rsplit("_", 1)[1])


def tgt(src):
    files = sorted(glob.glob(os.path.join(src, "ds", "*_*.npz")))
    for i, p in enumerate(files):
        D.targets(p, os.path.join(src, "raw", os.path.basename(p)), os.path.join(src, "tgt"))
        if i % 50 == 0:
            print(i, len(files), os.path.basename(p), flush=True)
    print("tgt done", len(files))


def link_set(work, name, picks):
    out = os.path.join(work, name)
    n = 0
    for sub in ("ds", "tgt"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    for src, base in picks:
        for sub in ("ds", "tgt"):
            s = os.path.join(src, sub, base)
            d = os.path.join(out, sub, base)
            if not os.path.isfile(s):
                raise SystemExit("missing " + s)
            if not os.path.exists(d):
                os.link(s, d)
        n += 1
    games = sorted({D.game_of(b) for _, b in picks})
    json.dump(dict(files=len(picks), games=len(games), pff=sum(g.startswith("PFF") for g in games),
                   dfl=sum(g.startswith("DFL") for g in games),
                   seeds=sorted({seed_of(b) for _, b in picks})), open(os.path.join(out, "config.json"), "w"), indent=1)
    print(name, n, "files,", len(games), "games")


def link(work):
    hold = {"PFF-" + g for g in D.holdout()}
    p2 = os.path.join(work, "ds_pff2")
    new = os.path.join(work, "ds_r4new")
    base = [(p2, os.path.basename(f)) for f in sorted(glob.glob(os.path.join(p2, "ds", "*_*.npz")))]
    assert len(base) == 165 and not any(D.game_of(b) in hold for _, b in base)
    nf = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(new, "ds", "*_*.npz")))]
    h_old = [(new, b) for b in nf if D.game_of(b) in hold and seed_of(b) in OLD]
    h_new = [(new, b) for b in nf if D.game_of(b) in hold and seed_of(b) in NEW]
    s_new = [(new, b) for b in nf if D.game_of(b) not in hold and seed_of(b) in NEW]
    assert len(h_old) == 48 and len(h_new) == 80 and len(s_new) == 48 * 5 + 7 * 5, (len(h_old), len(h_new), len(s_new))
    link_set(work, "ds_pff64", base + h_old)
    link_set(work, "ds_pff8", base + s_new)
    link_set(work, "ds_pff8_64", base + s_new + h_old + h_new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tgt", "link"])
    ap.add_argument("--work", default=D.WORK)
    ap.add_argument("--src", default=None)
    a = ap.parse_args()
    if a.what == "tgt":
        tgt(a.src or os.path.join(a.work, "ds_r4new"))
    else:
        link(a.work)


if __name__ == "__main__":
    main()
