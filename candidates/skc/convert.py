#!/usr/bin/env python3
"""SkillCorner A-League 2024/25 open data (20 games, MIT) -> the IDSSE npz format the DFL benches read.

    python -I candidates/skc/convert.py parse  [games...]      # raw jsonl -> source-rate cache (10 fps)
    python -I candidates/skc/convert.py write  [games...]      # cache + labels -> 25 fps npz
    python -I candidates/skc/convert.py all    [games...]      # both
    (--jobs 2 runs games in parallel; no games = all 20)

Input (never committed, read as data only): $MTAH_WORK/skc_aleague/raw = a git-lfs clone of
github.com/SkillCorner/opendata (data/matches/<id>/<id>_{tracking_extrapolated.jsonl, match.json,
dynamic_events.csv, phases_of_play.csv}). Output:

    $MTAH_WORK/skc_aleague/src/<game>.npz                    source-rate cache (10 fps)
    $MTAH_WORK/skc_aleague/npz/SKC-<game>.npz                chosen labels (labels.CHOSEN)
    $MTAH_WORK/skc_aleague/variants/<variant>/SKC-<game>.npz every label variant

The npz has the keys of the IDSSE npz (learned_dfl_evidence/fetch_idsse.py), so
bench.dfl2_prepared / bench_skc.skc2_prepared read it:

    players (N, P, 2) float32   metres, origin at a pitch corner (x in [0, 105], y in [0, 68]),
                                NaN when the person is not on the pitch
    ball    (N, 2) float32      metres, never NaN inside a period (gaps filled, see below)
    status  (N,) int8           1 alive, 0 dead
    poss    (N,) int8           1 home, 2 away
    half    (N,) int8           1, 2
    frame   (N,) int32          the SkillCorner frame (10 fps) nearest to the 25 fps sample
    team_of_player (P,) int8    0 home, 1 away
  plus SkillCorner extras: player_id (P,) int64; ball_skc (N, 2) the ball as given (NaN in broadcast
  cuts); ball_z (N,) float32; ball_det (N,) int8 (0 no data, 1 detected, 2 extrapolated by SkillCorner);
  video (N,) bool (False inside a broadcast cut: SkillCorner has no player and no ball data);
  ndet (N, 2) int8 players detected on screen per team (the others are SkillCorner's extrapolation);
  pitch (2,) float32 the stadium's length and width in metres.

Positions. SkillCorner's "extrapolated" tracking gives all 22 players on every frame that has data:
players off the broadcast picture are SkillCorner's own estimates (is_detected false), the analogue of
PFF's smoothed off-screen estimates, and are used as they are. 27-34% of in-period frames have no data
at all (broadcast cuts: replays, close-ups); ~97% of those are dead ball (outside every phase of play).
Inside a cut every player is linearly interpolated in time between his last position before and his
first after it, and the ball the same way (PFF's rule for its smoothed ball), so positions are
complete, like the PFF and DFL npz; `video` marks the filled rows. A player is only placed while
match.json's playing_time.sequences says he is on the pitch (substitutes and sent-off players are NaN
outside), with his first / last known position held at the edges of a sequence.

Coordinates: SkillCorner's origin is the centre spot, x along the touchlines, y along the goal lines,
on the stadium's own pitch (104-106 x 68 m). We map to the 105 x 68 corner-origin frame of the benches:
x' = (x + L/2) * 105 / L, y' = (y + W/2) * 68 / W (keeping the axes' signs, as fetch_idsse / PFF).

Rows: each period from match.json's start_frame to its end_frame, resampled from 10 to 25 fps on the
period's own clock (sample k at source frame start + k * 0.4). Positions are linearly interpolated
between the two neighbouring source frames; labels and flags take the nearest source frame (ties to
the earlier one).

Labels come from SkillCorner's own event and phase files (candidates/skc/labels.py), never by hand.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labels as L    # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                                   "external", "work")
BASE = os.path.join(WORK, "skc_aleague")
RAW = os.path.join(BASE, "raw")
SRC = os.path.join(BASE, "src")
OUT = os.path.join(BASE, "npz")
VAR = os.path.join(BASE, "variants")
FPS_SRC, FPS_OUT = 10.0, 25.0
PITCH = (105.0, 68.0)
GROUP = {"home team": 1, "away team": 2}


def games():
    d = os.path.join(RAW, "data", "matches")
    return sorted(g for g in os.listdir(d) if os.path.isfile(os.path.join(d, g, g + "_match.json")))


def meta(game):
    with open(os.path.join(RAW, "data", "matches", game, game + "_match.json")) as h:
        return json.load(h)


def _f(v):
    return float("nan") if v is None else float(v)


# ------------------------------------------------------------------------------------------------
# Stage 1: raw tracking -> source-rate cache
# ------------------------------------------------------------------------------------------------

def parse(game):
    m = meta(game)
    home, away = int(m["home_team"]["id"]), int(m["away_team"]["id"])
    col, team = {}, []
    for p in m["players"]:
        t = int(p["team_id"])
        if t in (home, away) and int(p["id"]) not in col:
            col[int(p["id"])] = len(team)
            team.append(0 if t == home else 1)
    path = os.path.join(RAW, "data", "matches", game, game + "_tracking_extrapolated.jsonl")
    recs = []
    with open(path) as h:
        for line in h:
            if line.strip():
                recs.append(json.loads(line))
    n = len(recs)
    frames = np.array([int(r["frame"]) for r in recs])
    assert np.array_equal(frames, np.arange(frames[0], frames[0] + n)), "frames not contiguous"
    assert frames[0] == 0, "first frame is not 0"
    P = len(team)
    players = np.full((n, P, 2), np.nan, np.float32)
    pdet = np.full((n, P), -1, np.int8)
    ball = np.full((n, 3), np.nan, np.float32)
    ball_det = np.zeros(n, np.int8)
    flag = np.zeros(n, np.int8)
    flag_pid = np.zeros(n, np.int64)
    period = np.zeros(n, np.int8)
    unknown = 0
    for i, r in enumerate(recs):
        period[i] = r.get("period") or 0
        for pl in r.get("player_data") or []:
            j = col.get(int(pl["player_id"]))
            if j is None:
                unknown += 1
                continue
            players[i, j] = (_f(pl.get("x")), _f(pl.get("y")))
            pdet[i, j] = 1 if pl.get("is_detected") else 0
        b = r.get("ball_data") or {}
        if b.get("x") is not None:
            ball[i] = (_f(b.get("x")), _f(b.get("y")), _f(b.get("z")))
            ball_det[i] = 1 if b.get("is_detected") else 2
        po = r.get("possession") or {}
        flag[i] = GROUP.get(po.get("group"), 0)
        flag_pid[i] = int(po.get("player_id") or 0)
    os.makedirs(SRC, exist_ok=True)
    np.savez_compressed(os.path.join(SRC, game + ".npz"), period=period, players=players, pdet=pdet,
                        ball=ball, ball_det=ball_det, flag=flag, flag_pid=flag_pid,
                        team_of_player=np.array(team, np.int8),
                        player_id=np.array(sorted(col, key=col.get), np.int64))
    return game, n, P, unknown


# ------------------------------------------------------------------------------------------------
# Stage 2: fill, labels (labels.py) and 25 fps resampling
# ------------------------------------------------------------------------------------------------

def on_pitch(m, col, n):
    """(n, P) bool: match.json playing_time.sequences (else by_period, else total), inclusive frames."""
    on = np.zeros((n, len(col)), bool)
    for p in m["players"]:
        j = col.get(int(p["id"]))
        pt = p.get("playing_time") or {}
        if j is None:
            continue
        seqs = pt.get("sequences") or pt.get("by_period") or ([pt["total"]] if pt.get("total") else [])
        for s in seqs:
            if s.get("start_frame") is not None and s.get("end_frame") is not None:
                on[int(s["start_frame"]):int(s["end_frame"]) + 1, j] = True
    return on


def fill_series(v, ok, idx):
    """Linear interpolation of v (1-D) at idx from the rows where ok (edges held); NaN if none."""
    if ok.sum() == 0:
        return np.full(len(idx), np.nan)
    k = np.where(ok)[0]
    return np.interp(idx, k, v[k])


def fill_period(src, on, a0, b0):
    """Players and ball of one period at the source rate, cuts filled."""
    sl = slice(a0, b0 + 1)
    pl = src["players"][sl].astype(np.float64)
    bl = src["ball"][sl, :2].astype(np.float64)
    idx = np.arange(b0 - a0 + 1)
    out = np.full_like(pl, np.nan)
    for j in range(pl.shape[1]):
        o = on[sl, j]
        if not o.any():
            continue
        # fill each contiguous on-pitch stretch separately (a player may leave and come back)
        d = np.diff(np.r_[0, o.astype(int), 0])
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            ok = np.isfinite(pl[s:e, j, 0])
            for c in range(2):
                out[s:e, j, c] = fill_series(pl[s:e, j, c], ok, idx[s:e] - s)
    ok = np.isfinite(bl[:, 0])
    ball = np.stack([fill_series(bl[:, c], ok, idx) for c in range(2)], 1)
    return out, ball, bl


def to_corner(xy, pitch_len, pitch_wid):
    out = np.array(xy, np.float64, copy=True)
    out[..., 0] = (out[..., 0] + pitch_len / 2.0) * PITCH[0] / pitch_len
    out[..., 1] = (out[..., 1] + pitch_wid / 2.0) * PITCH[1] / pitch_wid
    return out


def lin(arr, t):
    """Linear interpolation of arr rows at fractional row t; where a neighbour is NaN use the other."""
    j = np.clip(np.floor(t).astype(int), 0, len(arr) - 1)
    k = np.clip(j + 1, 0, len(arr) - 1)
    w = (t - j).reshape((-1,) + (1,) * (arr.ndim - 1))
    a, b = arr[j], arr[k]
    out = (1 - w) * a + w * b
    out = np.where(np.isnan(a), b, out)
    return np.where(np.isnan(b), a, out)


def write(game):
    z = np.load(os.path.join(SRC, game + ".npz"))
    src = {k: z[k] for k in z.files}
    m = meta(game)
    n = len(src["period"])
    col = {int(p): j for j, p in enumerate(src["player_id"])}
    on = on_pitch(m, col, n)
    variants = L.build_all(RAW, game, n, src["flag"])
    Lp, Wp = float(m.get("pitch_length") or 105), float(m.get("pitch_width") or 68)
    team = src["team_of_player"]
    parts = {k: [] for k in ("players", "ball", "ball_skc", "ball_z", "ball_det", "video", "ndet", "half",
                             "frame")}
    for per, (a0, b0) in sorted(L.periods(m, n).items()):
        pl, bl, braw = fill_period(src, on, a0, b0)
        t = np.arange(int(np.floor((b0 - a0) * FPS_OUT / FPS_SRC)) + 1) * FPS_SRC / FPS_OUT
        near = a0 + np.floor(t + 0.5 - 1e-9).astype(int)
        near = np.clip(near, a0, b0)
        parts["players"].append(to_corner(lin(pl, t), Lp, Wp))
        parts["ball"].append(to_corner(lin(bl, t), Lp, Wp))
        parts["ball_skc"].append(to_corner(lin(braw, t), Lp, Wp))
        parts["ball_z"].append(lin(src["ball"][a0:b0 + 1, 2].astype(np.float64), t))
        parts["ball_det"].append(src["ball_det"][near])
        det = src["pdet"][near]
        parts["video"].append((det >= 0).any(1))
        parts["ndet"].append(np.stack([((det == 1) & (team[None] == s)).sum(1) for s in (0, 1)], 1))
        parts["half"].append(np.full(len(t), per, np.int8))
        parts["frame"].append(near)
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    near = cat["frame"]
    used = np.isfinite(cat["players"][..., 0]).any(0)        # drop squad members never on the pitch
    assert used.sum() >= 22, (game, int(used.sum()), "players on the pitch")
    base = dict(players=cat["players"][:, used].astype(np.float32), ball=cat["ball"].astype(np.float32),
                ball_skc=cat["ball_skc"].astype(np.float32), ball_z=cat["ball_z"].astype(np.float32),
                ball_det=cat["ball_det"].astype(np.int8), video=cat["video"], ndet=cat["ndet"].astype(np.int8),
                half=cat["half"], frame=near.astype(np.int32), team_of_player=team[used].astype(np.int8),
                player_id=src["player_id"][used], pitch=np.float32([Lp, Wp]))
    assert np.isfinite(base["ball"]).all()
    for name, (status, poss) in variants.items():
        st, po = status[near].astype(np.int8), poss[near].astype(np.int8)
        assert ((po > 0) | (st == 0)).all(), (game, name, "alive frame without a team")
        d = os.path.join(VAR, name)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "SKC-" + game + ".npz"), status=st, poss=po, **base)
        if name == L.CHOSEN:
            os.makedirs(OUT, exist_ok=True)
            np.savez_compressed(os.path.join(OUT, "SKC-" + game + ".npz"), status=st, poss=po, **base)
    return game, len(near), int(used.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["parse", "write", "all"])
    ap.add_argument("games", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    todo = a.games or games()
    fns = {"parse": [parse], "write": [write], "all": [parse, write]}[a.stage]
    for fn in fns:
        if a.jobs > 1:
            with ProcessPoolExecutor(a.jobs) as ex:
                for r in ex.map(fn, todo):
                    print(fn.__name__, r, flush=True)
        else:
            for g in todo:
                print(fn.__name__, fn(g), flush=True)


if __name__ == "__main__":
    main()
