#!/usr/bin/env python3
"""Player-level possessor targets for the possessor-pointer model (round 5, R5-3).

    python -I candidates/nn_pointer/make_owner.py --ds $MTAH_WORK/ds_pff2 [--jobs 3]

Step 1, per game (cached in $MTAH_WORK/arch/r5_arch/possessor/<game>.npz): pcol (N,) int16, the clean
tracking column of the player on the ball at every 25 fps row, -1 = unknown.
  PFF  the player of the latest on-the-ball game event (OTB / kick-off) of the period, i.e. exactly
       the player behind the "touch" team label (candidates/pff/labels.py); PFF player_id -> roster
       (team, shirt) -> source column -> npz column (matched by position). Kept only where his team
       equals the label's team.
  DFL  no usable player-level stream (the event clock jitters by ~1 s), so a proxy: the last
       "controller" (nearest player within 1 m of the clean, tracked ball) of the flag's team inside
       the current team spell; -1 until that team's first controller of the spell.
Step 2, per ds file (one degradation seed, one stride-5 phase): own (T,) int16 at the file's idx rows,
<ds>/owner/<name>_<seed>.npz: the degraded slot (0..S-1) nearest to the possessor's clean position among
the slots of his team (<= 3 m), -1 unknown (no possessor or no slot close enough), -2 dead (code 2).
Targets only; the model's inputs are the degraded slots.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np

WORK = os.environ.get("MTAH_WORK")
PCACHE = os.path.join(WORK or "", "arch", "r5_arch", "possessor")
ON_BALL = ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF")
CTRL_R, SLOT_R = 1.0, 3.0


def codes_of(z):
    st, po = z["status"], z["poss"]
    c = np.full(len(st), 3, np.int8)
    c[(st == 1) & (po == 1)] = 0
    c[(st == 1) & (po == 2)] = 1
    c[st == 0] = 2
    return c


def pff_pcol(game):
    gid = game[4:]
    pw = os.path.join(WORK, "pff_wc2022")
    z = np.load(os.path.join(pw, "npz", game + ".npz"))
    src = np.load(os.path.join(pw, "src", gid + ".npz"))
    ev = json.load(open(os.path.join(pw, "src", gid + "_events.json")))
    roster = json.load(open(os.path.join(pw, "raw", "Rosters", gid + ".json")))
    meta = json.load(open(os.path.join(pw, "raw", "Metadata", gid + ".json")))
    meta = meta[0] if isinstance(meta, list) else meta
    home_id = str(meta["homeTeam"]["id"])
    pid2key = {str(r["player"]["id"]): (0 if str(r["team"]["id"]) == home_id else 1, str(r["shirtNumber"])) for r in roster}
    key2src = {(int(t), str(j)): k for k, (t, j) in enumerate(zip(src["team_of_player"], src["jersey"]))}
    # npz column <- source column, by position (the npz keeps the columns ever on the pitch, interpolated)
    sfr = src["frame"]
    rows = np.searchsorted(sfr, z["frame"])
    rows = np.clip(rows, 0, len(sfr) - 1)
    P, SP = z["players"], src["players"] + np.float32([52.5, 34.0])
    samp = np.linspace(0, len(rows) - 1, 4000).astype(int)
    npz2src = np.full(P.shape[1], -1)
    for j in range(P.shape[1]):
        a = P[samp, j]
        cand = np.where(src["team_of_player"] == z["team_of_player"][j])[0]
        best, bk = np.inf, -1
        for k in cand:
            d = np.linalg.norm(a - SP[rows[samp], k], axis=1)
            ok = ~np.isnan(d)
            if ok.sum() < 20 or (~np.isnan(a[:, 0])).sum() > 1.5 * ok.sum():
                continue
            m = float(np.mean(d[ok]))
            if m < best:
                best, bk = m, k
        if best < 1.0:
            npz2src[j] = bk
    src2npz = {int(k): j for j, k in enumerate(npz2src) if k >= 0}
    codes = codes_of(z)
    pcol = np.full(len(codes), -1, np.int16)
    for per in (1, 2):
        sel = np.where(z["half"] == per)[0]
        f = z["frame"][sel]
        obs = sorted((e for e in ev["game_events"] if e["type"] in ON_BALL and e["period"] == per
                      and e["start_frame"] is not None), key=lambda e: (int(e["start_frame"]), e["id"]))
        for i, e in enumerate(obs):
            key = pid2key.get(str(e.get("player_id")))
            col = src2npz.get(key2src.get(key, -1), -1) if key else -1
            a = int(e["start_frame"])
            b = int(obs[i + 1]["start_frame"]) if i + 1 < len(obs) else 1 << 40
            pcol[sel[(f >= a) & (f < b)]] = col
    team = np.where(pcol >= 0, z["team_of_player"][np.maximum(pcol, 0)], -1)
    pcol[(codes > 1) | (team != codes)] = -1
    return pcol, dict(matched=int((npz2src >= 0).sum()), cols=int(P.shape[1]))


def dfl_pcol(game):
    z = np.load(os.path.join(WORK, "idsse", "npz", game + ".npz"))
    P, B, tp = z["players"].astype(np.float64), z["ball"].astype(np.float64), z["team_of_player"]
    codes = codes_of(z)
    d = np.linalg.norm(P - B[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, 1)
    ctrl = np.where(d[np.arange(len(j)), j] < CTRL_R, j, -1)
    pcol = np.full(len(codes), -1, np.int16)
    cur, prev = -1, -9
    for t in range(len(codes)):
        c = codes[t]
        if c != prev:
            cur = -1
            prev = c
        if c > 1:
            continue
        if ctrl[t] >= 0 and tp[ctrl[t]] == c:
            cur = ctrl[t]
        pcol[t] = cur
    return pcol, dict(cols=int(P.shape[1]))


def possessor(game):
    os.makedirs(PCACHE, exist_ok=True)
    p = os.path.join(PCACHE, game + ".npz")
    if os.path.isfile(p):
        return np.load(p)["pcol"]
    pcol, info = pff_pcol(game) if game.startswith("PFF-") else dfl_pcol(game)
    np.savez(p, pcol=pcol, info=json.dumps(info))
    return pcol


def src_path(game):
    return os.path.join(WORK, "pff_wc2022", "npz", game + ".npz") if game.startswith("PFF-") else \
        os.path.join(WORK, "idsse", "npz", game + ".npz")


def build(ds_path, out_path):
    if os.path.isfile(out_path):
        return "exists " + out_path
    name = os.path.basename(ds_path)[:-4].rsplit("_", 1)[0]
    pcol = possessor(name)
    idx = np.load(ds_path)["idx"].astype(np.int64)
    raw = np.load(os.path.join(os.path.dirname(os.path.dirname(ds_path)), "raw", os.path.basename(ds_path)))
    slots, st, codes = raw["slots"], raw["slot_team"], raw["codes"]
    g = np.load(src_path(name))
    P, tp = g["players"], g["team_of_player"]
    own = np.full(len(idx), -1, np.int16)
    c = codes[idx]
    own[c == 2] = -2
    pc = pcol[idx]
    k = np.where((pc >= 0) & (c < 2))[0]
    r = idx[k]
    pos = P[r, pc[k]]                                     # (K, 2) clean possessor
    d = np.linalg.norm(slots[r] - pos[:, None, :], axis=2)  # (K, S)
    d = np.where(st[None, :] == tp[pc[k]][:, None], d, np.inf)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, 1)
    ok = d[np.arange(len(k)), j] <= SLOT_R
    own[k[ok]] = j[ok]
    tmp = out_path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, own=own, idx=idx)
    os.replace(tmp, out_path)
    alive = c < 2
    return "{} alive {} possessor known {:.2f} slot found {:.2f}".format(
        os.path.basename(out_path), int(alive.sum()), float((pc[alive] >= 0).mean()), float((own[alive] >= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.ds, "owner"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.ds, "ds", "*_*.npz")))
    games = sorted({os.path.basename(p)[:-4].rsplit("_", 1)[0] for p in files})
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        pool.map(possessor, games)
        for m in pool.starmap(build, [(p, os.path.join(a.ds, "owner", os.path.basename(p))) for p in files]):
            print(m, flush=True)


if __name__ == "__main__":
    main()
