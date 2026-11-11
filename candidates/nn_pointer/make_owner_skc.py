#!/usr/bin/env python3
"""Possessor-pointer targets for SkillCorner A-League ds files (round-5 "+SKC" experiment; make_owner.py extended).

    python -I candidates/nn_pointer/make_owner_skc.py --ds $MTAH_WORK/ds_pff2_skc20 [--jobs 8]

Builds <ds>/owner/<file> for the SKC-* ds files only (the PFF / DFL files are make_owner.py's, untouched), with
make_owner.build (same slot matching: the degraded slot of the possessor's team nearest his clean position, <= 3 m;
-1 unknown, -2 dead). Step 1 for SKC = the PFF rule on SkillCorner's own events: the player of the latest
player_possession (from its frame_start) or of an on-ball engagement ending direct_disruption / direct_regain (from its
frame_end; a possession start on the same frame wins), i.e. exactly the player behind the touch_eng team label
(candidates/skc/labels.py); kept only where his team equals the label's team. Event player_id -> npz column through the
npz's player_id. Cache: $MTAH_WORK/arch/r5_arch/skc_train/possessor/<game>.npz (not make_owner's cache folder).
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_owner as MO  # noqa: E402

WORK = os.environ["MTAH_WORK"]
SKC = os.path.join(WORK, "skc_aleague")
PCACHE = os.path.join(WORK, "arch", "r5_arch", "skc_train", "possessor")
DIRECT = ("direct_disruption", "direct_regain")


def skc_pcol(game):
    gid = game[4:]
    z = np.load(os.path.join(SKC, "npz", game + ".npz"))
    ev = pd.read_csv(os.path.join(SKC, "raw", "data", "matches", gid, gid + "_dynamic_events.csv"), low_memory=False)
    col_of = {int(p): j for j, p in enumerate(z["player_id"])}
    pp = ev[ev.event_type == "player_possession"]
    ob = ev[(ev.event_type == "on_ball_engagement") & ev.end_type.isin(DIRECT)]
    ch = [(int(f), 1, int(per), col_of.get(int(p), -1)) for f, per, p in zip(pp.frame_start, pp.period, pp.player_id)]
    ch += [(int(f), 0, int(per), col_of.get(int(p), -1)) for f, per, p in zip(ob.frame_end, ob.period, ob.player_id)]
    ch.sort()
    codes = MO.codes_of(z)
    pcol = np.full(len(codes), -1, np.int16)
    for per in (1, 2):
        sel = np.where(z["half"] == per)[0]
        c = [(f, col) for f, _, p, col in ch if p == per]
        if not len(sel) or not c:
            continue
        fr = np.array([f for f, _ in c])
        cols = np.array([col for _, col in c])
        # latest change at or before the row's source frame (ties: the last in sorted order = possession start)
        k = np.searchsorted(fr, z["frame"][sel], side="right") - 1
        pcol[sel] = np.where(k >= 0, cols[np.maximum(k, 0)], -1)
    team = np.where(pcol >= 0, z["team_of_player"][np.maximum(pcol, 0)], -1)
    pcol[(codes > 1) | (team != codes)] = -1
    return pcol, dict(cols=int(z["players"].shape[1]), unmatched_players=int(sum(c[3] < 0 for c in ch)))


def possessor(game):
    if not game.startswith("SKC-"):
        raise ValueError(game)
    os.makedirs(PCACHE, exist_ok=True)
    p = os.path.join(PCACHE, game + ".npz")
    if os.path.isfile(p):
        return np.load(p)["pcol"]
    pcol, info = skc_pcol(game)
    tmp = p[:-4] + ".tmp{}.npz".format(os.getpid())
    np.savez(tmp, pcol=pcol, info=json.dumps(info))
    os.replace(tmp, p)
    return pcol


def src_path(game):
    return os.path.join(SKC, "npz", game + ".npz")


# make_owner.build looks these up as module globals: point them at the SkillCorner versions (this process and the
# spawn workers, which import this file as their main module)
MO.possessor = possessor
MO.src_path = src_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.ds, "owner"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.ds, "ds", "SKC-*_*.npz")))
    games = sorted({os.path.basename(p)[:-4].rsplit("_", 1)[0] for p in files})
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        pool.map(possessor, games)
        for m in pool.starmap(MO.build, [(p, os.path.join(a.ds, "owner", os.path.basename(p))) for p in files]):
            print(m, flush=True)


if __name__ == "__main__":
    main()
