#!/usr/bin/env python3
"""Parse the SkillCorner open data into the DFL npz layout at 25 fps, with algorithmic H/A/D labels.

    python -I parse_skillcorner.py --raw DIR --out DIR

Per match -> <out>/SC-<id>.npz with players (N, P, 2) metres in [0, 105] x [0, 68] (NaN where the
player is not DETECTED on the broadcast image; SkillCorner's extrapolated off-screen positions are
kept separately in players_ex), ball (N, 2) (NaN unless the ball is detected), poss (1 home /
2 away / 0), status (1 alive / 0 dead), half, frame (source 10 fps frame), team_of_player.

Labels (mechanical, no smoothing, nothing hand-made):
    status = 1 where the tracking frame's possession.group is "home team" / "away team",
             0 where it is null, inside the match periods (match.json match_periods).
    poss   = 1 / 2 from possession.group.
SkillCorner's possession.group is a team possession that persists across passes and loose balls
and is null while the ball is out of play (it agrees with the Phases of Play windows, which are
only defined while the ball is in play, on ~95% of frames; the ball is outside the pitch on
0.1-0.4% of group frames but on 12-16% of null frames). That is the same meaning as DFL's
BallStatus / BallPossession flags and the competition ground truth ("D where the tracking marks
the ball dead, otherwise the team the tracking names").

Resampling 10 -> 25 fps: target frame k sits at source position 0.4 k; positions are linearly
interpolated between the two neighbouring source frames when both are detected (NaN otherwise,
except on exact source frames); labels take the nearest source frame. Coordinates are shifted
from SkillCorner's centre origin and scaled from the match's pitch size to 105 x 68 m.
"""

import argparse
import glob
import json
import os

import numpy as np

L0, W0 = 105.0, 68.0
GROUP = {"home team": 1, "away team": 2}


def parse(folder):
    mid = os.path.basename(folder.rstrip("/"))
    meta = json.load(open(os.path.join(folder, "{}_match.json".format(mid))))
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    pids = sorted(p["id"] for p in meta["players"])
    col = {p: i for i, p in enumerate(pids)}
    team = np.array([0 if next(q for q in meta["players"] if q["id"] == p)["team_id"] == home_id
                     else 1 for p in pids], np.int8)
    rows = [json.loads(l) for l in open(os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(mid)))]
    n = len(rows)
    P = len(pids)
    pl = np.full((n, P, 2), np.nan, np.float32)
    plx = np.full((n, P, 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    grp = np.zeros(n, np.int8)
    per = np.zeros(n, np.int8)
    for i, r in enumerate(rows):
        per[i] = r["period"] or 0
        grp[i] = GROUP.get(r["possession"]["group"], 0)
        b = r["ball_data"]
        if b["is_detected"] is True and b["x"] is not None:
            ball[i] = (b["x"], b["y"])
        for p in r["player_data"]:
            j = col.get(p["player_id"])
            if j is None:
                continue
            plx[i, j] = (p["x"], p["y"])
            if p["is_detected"]:
                pl[i, j] = (p["x"], p["y"])
    # only frames inside the match periods are graded
    inper = np.zeros(n, bool)
    for mp in meta["match_periods"]:
        inper[mp["start_frame"]:mp["end_frame"] + 1] = True
    per[~inper] = 0

    def to_pitch(a):
        out = a.copy()
        out[..., 0] = (a[..., 0] + L / 2) * (L0 / L)
        out[..., 1] = (a[..., 1] + W / 2) * (W0 / W)
        return out
    pl, plx, ball = to_pitch(pl), to_pitch(plx), to_pitch(ball)

    # 10 -> 25 fps
    m = int(np.floor((n - 1) * 2.5)) + 1
    pos = np.arange(m) * 0.4
    lo = np.floor(pos + 1e-9).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo).astype(np.float32)
    exact = w < 1e-6

    def interp(a):
        wa = w.reshape((-1,) + (1,) * (a.ndim - 1))
        v = a[lo] * (1 - wa) + a[hi] * wa
        return np.where(exact.reshape(wa.shape), a[lo], v).astype(np.float32)
    near = np.minimum(np.round(pos).astype(int), n - 1)
    g = grp[near]
    half = per[near]
    status = np.where(g > 0, 1, 0).astype(np.int8)
    status[half == 0] = 0
    out = dict(players=interp(pl), players_ex=interp(plx), ball=interp(ball), poss=g.astype(np.int8),
               status=status, half=half.astype(np.int8), frame=near.astype(np.int32),
               team_of_player=team, pitch_src=np.array([L, W]))
    return mid, out


def label_stats(z):
    graded = z["half"] > 0
    codes = np.where(z["status"] == 0, 2, z["poss"] - 1)
    codes[~graded] = 3
    spells, durs = 0, []
    for h in (1, 2):
        c = codes[z["half"] == h]
        brk = np.flatnonzero(np.diff(c) != 0) + 1
        s, e = np.r_[0, brk], np.r_[brk, len(c)]
        al = c[s] < 2
        spells += int(al.sum())
        durs += list((e - s)[al] / 25.0)
    mins = graded.sum() / 25 / 60
    return dict(minutes=round(float(mins), 1), spells=spells,
                spells_per90=round(spells / mins * 90, 1),
                dead_share=round(float((codes[graded] == 2).mean()), 3),
                dur_p10_50_90=[round(float(v), 1) for v in np.percentile(durs, [10, 50, 90])],
                short_lt1s=round(float(np.mean(np.array(durs) < 1)), 3),
                ball_missing=round(float(np.isnan(z["ball"][graded, 0]).mean()), 3),
                ball_missing_alive=round(float(np.isnan(z["ball"][graded & (codes < 2), 0]).mean()), 3),
                players_detected=round(float((~np.isnan(z["players"][graded, :, 0])).sum(1).mean()), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stats = {}
    for folder in sorted(glob.glob(os.path.join(args.raw, "[0-9]*"))):
        path = os.path.join(args.out, "SC-{}.npz".format(os.path.basename(folder)))
        if not os.path.isfile(path):
            mid, z = parse(folder)
            np.savez_compressed(path, **z)
        z = dict(np.load(path))
        stats[os.path.basename(path)[:-4]] = s = label_stats(z)
        print(os.path.basename(path), s, flush=True)
    tot = sum(s["spells"] for s in stats.values())
    mins = sum(s["minutes"] for s in stats.values())
    print("ALL spells", tot, "per90", round(tot / mins * 90, 1), "minutes", round(mins, 1),
          "dead share", round(float(np.mean([s["dead_share"] for s in stats.values()])), 3))
    json.dump(stats, open(os.path.join(args.out, "label_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
