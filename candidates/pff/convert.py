#!/usr/bin/env python3
"""PFF FC World Cup 2022 (64 games) -> the IDSSE npz format the DFL benches read.

    python -I candidates/pff/convert.py parse  [games...]      # raw jsonl.bz2 -> source-rate cache
    python -I candidates/pff/convert.py write  [games...]      # cache + labels -> 25 fps npz
    python -I candidates/pff/convert.py all    [games...]      # both
    (--jobs 2 runs games in parallel; no games = all 64)

Input (never committed, read as data only): $MTAH_WORK/pff_wc2022/raw/{Tracking Data, Event Data,
Metadata, Rosters}. Output:

    $MTAH_WORK/pff_wc2022/src/<game>.npz + <game>_events.json   source-rate cache (29.97 fps)
    $MTAH_WORK/pff_wc2022/npz/PFF-<game>.npz                     chosen labels (labels.CHOSEN)
    $MTAH_WORK/pff_wc2022/variants/<variant>/PFF-<game>.npz      every label variant

The npz has the keys of the IDSSE npz (learned_dfl_evidence/fetch_idsse.py), so
bench.dfl2_prepared reads it with BENCH_DFL_DIR pointed at the folder:

    players (N, P, 2) float32   metres, origin at a pitch corner (x in [0, 105], y in [0, 68]),
                                NaN when the person is not on the pitch
    ball    (N, 2) float32      metres, never NaN: PFF's smoothed ball, gaps filled (see below)
    status  (N,) int8           1 alive, 0 dead
    poss    (N,) int8           1 home, 2 away (0 never occurs on graded frames)
    half    (N,) int8           1, 2 (extra time 3, 4 is dropped)
    frame   (N,) int32          the PFF frameNum nearest to the 25 fps sample
    team_of_player (P,) int8    0 home, 1 away
  plus PFF-only extras: ball_pff (N, 2) the smoothed ball as PFF gives it (NaN in gaps), ball_vis (N,) int8 (0 none, 1 VISIBLE, 2 ESTIMATED; the smoothed ball's
  visibility), ball_raw_vis (raw broadcast ball), nvis (N, 2) int8 (players visible on the
  broadcast per team), video (N,) bool (False inside a broadcast cut: no raw players and no ball).

Positions are PFF's *Smoothed* tracks (homePlayersSmoothed / awayPlayersSmoothed: Kalman-filtered,
all on-pitch players including off-screen estimates; ballsSmoothed: the ball placed at the carrier
by PFF's own events, linearly interpolated between events). They are the "clean" input that the
benches then degrade (degrade_v2). PFF gives no ball at all while play is stopped (the smoothed
ball is missing on ~99% of dead frames and ~6% of live ones), so `ball` fills every gap of a period
by linear interpolation between the last and the next known positions, which is PFF's own rule for
the smoothed ball between events. Without the fill a missing ball would be a perfect dead-ball
label; with it the ball is always present, as in the DFL npz, and degrade_v2 decides what the
candidate sees (it hides the ball over ~78% of dead runs, as on DFL). The coordinates are centred in PFF (x along the touchlines, y
along the goal lines); like fetch_idsse.py for DFL we add (52.5, 34) and keep the axes' signs.
All 64 stadiums are 105 x 68.

Rows: each regulation period from its kick-off frame (FIRSTKICKOFF / SECONDKICKOFF game event)
to its last tracking frame, resampled from 29.97 fps to 25 fps on the period's own clock:
sample k sits at source position kick + k * fps_src / 25. Positions are linearly interpolated
between the two neighbouring source frames (the nearer one if the other is missing); labels and the
visibility flags take the nearest source frame. Source frames listed several times (one copy per
possession event attached to the frame) are de-duplicated, keeping each player's first entry.

Labels come from PFF's own events (candidates/pff/labels.py), never from hand annotation.
"""

import argparse
import bz2
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
BASE = os.path.join(WORK, "pff_wc2022")
RAW = os.path.join(BASE, "raw")
SRC = os.path.join(BASE, "src")
OUT = os.path.join(BASE, "npz")
VAR = os.path.join(BASE, "variants")
FPS_OUT = 25.0
VIS = {None: 0, "VISIBLE": 1, "ESTIMATED": 2}


def games():
    d = os.path.join(RAW, "Tracking Data")
    return sorted(f.split(".")[0] for f in os.listdir(d) if f.endswith(".jsonl.bz2"))


def meta(game):
    with open(os.path.join(RAW, "Metadata", game + ".json")) as h:
        m = json.load(h)
    return m[0] if isinstance(m, list) else m


def roster(game):
    with open(os.path.join(RAW, "Rosters", game + ".json")) as h:
        return json.load(h)


# ------------------------------------------------------------------------------------------------
# Stage 1: raw tracking -> source-rate cache
# ------------------------------------------------------------------------------------------------

def _fnum(v):
    return float("nan") if v is None else float(v)


def parse(game):
    """One pass over the tracking file. Keeps periods 1 and 2; frames are de-duplicated."""
    m = meta(game)
    home_id, away_id = str(m["homeTeam"]["id"]), str(m["awayTeam"]["id"])
    persons = {}                                    # (side, jersey) -> column
    for r in roster(game):
        side = 0 if str(r["team"]["id"]) == home_id else 1 if str(r["team"]["id"]) == away_id else None
        if side is not None:
            persons.setdefault((side, str(r["shirtNumber"])), len(persons))
    rows, idx = [], {}
    gevents, pevents = {}, {}
    path = os.path.join(RAW, "Tracking Data", game + ".jsonl.bz2")
    with bz2.open(path, "rt") as handle:
        for line in handle:
            f = json.loads(line)
            per = f.get("period")
            ge, pe = f.get("game_event"), f.get("possession_event")
            if ge and f.get("game_event_id") is not None:
                gid = int(f["game_event_id"])
                if gid not in gevents:
                    gevents[gid] = dict(id=gid, type=ge.get("game_event_type"),
                                        start_frame=ge.get("start_frame"), end_frame=ge.get("end_frame"),
                                        home_team=ge.get("home_team"), home_ball=ge.get("home_ball"),
                                        sequence=ge.get("sequence"), player_id=ge.get("player_id"),
                                        team_id=ge.get("team_id"), period=per,
                                        start_time=ge.get("start_time"))
            if pe and f.get("possession_event_id") is not None:
                pid = int(f["possession_event_id"])
                if pid not in pevents:
                    pevents[pid] = dict(id=pid, game_event_id=pe.get("game_event_id"),
                                        type=pe.get("possession_event_type"),
                                        start_frame=pe.get("start_frame"), end_frame=pe.get("end_frame"),
                                        period=per)
            if per not in (1, 2):
                continue
            fn = int(f["frameNum"])
            if fn in idx:
                continue                            # duplicate copy of a frame
            idx[fn] = len(rows)
            pos = {}
            for side, key in ((0, "homePlayersSmoothed"), (1, "awayPlayersSmoothed")):
                for pl in f.get(key) or []:
                    k = (side, str(pl.get("jerseyNum")))
                    if k not in persons:
                        persons[k] = len(persons)
                    if persons[k] not in pos and pl.get("x") is not None:
                        pos[persons[k]] = (float(pl["x"]), float(pl["y"]))
            nvis = [sum(1 for pl in (f.get(k) or []) if pl.get("visibility") == "VISIBLE")
                    for k in ("homePlayers", "awayPlayers")]
            nraw = len(f.get("homePlayers") or []) + len(f.get("awayPlayers") or [])
            bs = f.get("ballsSmoothed")
            if isinstance(bs, list):
                bs = bs[0] if bs else None
            braw = (f.get("balls") or [None])[0]
            rows.append((fn, per, pos,
                         (_fnum(bs.get("x")), _fnum(bs.get("y"))) if bs else (np.nan, np.nan),
                         VIS.get(bs.get("visibility"), 0) if bs and bs.get("x") is not None else 0,
                         (_fnum(braw.get("x")), _fnum(braw.get("y")), _fnum(braw.get("z"))) if braw else (np.nan,) * 3,
                         VIS.get(braw.get("visibility"), 0) if braw and braw.get("x") is not None else 0,
                         nvis, nraw, float(f.get("videoTimeMs") or np.nan)))
    rows.sort(key=lambda r: r[0])
    n, P = len(rows), len(persons)
    players = np.full((n, P, 2), np.nan, np.float32)
    for i, r in enumerate(rows):
        for j, xy in r[2].items():
            players[i, j] = xy
    team = np.zeros(P, np.int8)
    for (side, _), j in persons.items():
        team[j] = side
    os.makedirs(SRC, exist_ok=True)
    np.savez_compressed(os.path.join(SRC, game + ".npz"),
                        frame=np.array([r[0] for r in rows], np.int32),
                        period=np.array([r[1] for r in rows], np.int8),
                        players=players, team_of_player=team,
                        jersey=np.array([k[1] for k, _ in sorted(persons.items(), key=lambda t: t[1])]),
                        ball=np.array([r[3] for r in rows], np.float32),
                        ball_vis=np.array([r[4] for r in rows], np.int8),
                        ball_raw=np.array([r[5] for r in rows], np.float32),
                        ball_raw_vis=np.array([r[6] for r in rows], np.int8),
                        nvis=np.array([r[7] for r in rows], np.int8),
                        nraw=np.array([r[8] for r in rows], np.int16),
                        video_ms=np.array([r[9] for r in rows], np.float64))
    with open(os.path.join(SRC, game + "_events.json"), "w") as h:
        json.dump(dict(game_events=sorted(gevents.values(), key=lambda e: (e["start_frame"] or 0, e["id"])),
                       possession_events=sorted(pevents.values(), key=lambda e: (e["start_frame"] or 0, e["id"]))), h)
    return game, n, P


def load_src(game):
    z = np.load(os.path.join(SRC, game + ".npz"))
    with open(os.path.join(SRC, game + "_events.json")) as h:
        ev = json.load(h)
    return {k: z[k] for k in z.files}, ev


# ------------------------------------------------------------------------------------------------
# Stage 2: labels (labels.py) and 25 fps resampling
# ------------------------------------------------------------------------------------------------

def resample_index(frames, kick, last, fps_src):
    """Source positions (float, in frameNum units) of the 25 fps samples of one period."""
    n = int(np.floor((last - kick) * FPS_OUT / fps_src)) + 1
    return kick + np.arange(n) * fps_src / FPS_OUT


def interp_rows(arr, frames, t):
    """Linear interpolation of arr (rows indexed by sorted frames) at fractional frame t; where a
    neighbour is NaN, the other (if finite) is used."""
    j = np.clip(np.searchsorted(frames, t, side="right") - 1, 0, len(frames) - 1)
    k = np.clip(j + 1, 0, len(frames) - 1)
    fj, fk = frames[j].astype(np.float64), frames[k].astype(np.float64)
    w = np.where(fk > fj, (t - fj) / np.maximum(fk - fj, 1e-9), 0.0)
    w = np.clip(w, 0.0, 1.0).reshape((-1,) + (1,) * (arr.ndim - 1))
    a, b = arr[j].astype(np.float64), arr[k].astype(np.float64)
    out = (1 - w) * a + w * b
    out = np.where(np.isnan(a), b, out)
    out = np.where(np.isnan(b), a, out)
    return out


def fill_gaps(xy, frames):
    """PFF's smoothed ball with every gap of one period filled by linear interpolation in time
    between the last and next known positions (held at the period's edges)."""
    out = xy.astype(np.float64).copy()
    ok = np.isfinite(xy[:, 0])
    if ok.sum() >= 2:
        for c in range(2):
            out[:, c] = np.interp(frames, frames[ok], xy[ok, c])
    return out


def nearest_rows(frames, t):
    j = np.clip(np.searchsorted(frames, t), 0, len(frames) - 1)
    jm = np.clip(j - 1, 0, len(frames) - 1)
    return np.where(np.abs(frames[jm] - t) <= np.abs(frames[j] - t), jm, j)


def write(game):
    src, ev = load_src(game)
    m = meta(game)
    fps_src = float(m.get("fps") or 29.97)
    frames = src["frame"]
    variants = L.build_all(game, src, ev, fps_src, RAW)     # name -> (status, poss) at source rate
    kicks = L.kickoffs(ev)
    parts = {k: [] for k in ("players", "ball", "ball_pff", "half", "frame", "ball_vis", "ball_raw_vis", "nvis",
                             "video", "src")}
    for per in (1, 2):
        sel = np.where(src["period"] == per)[0]
        kick, last = kicks[per], int(frames[sel[-1]])
        t = resample_index(frames, kick, last, fps_src)
        f = frames[sel]
        parts["players"].append(interp_rows(src["players"][sel] + np.float32([52.5, 34.0]), f, t))
        bp = src["ball"][sel] + np.float32([52.5, 34.0])
        parts["ball_pff"].append(interp_rows(bp, f, t))
        parts["ball"].append(interp_rows(fill_gaps(bp, f), f, t))
        nr = sel[nearest_rows(f, t)]
        parts["src"].append(nr)
        parts["half"].append(np.full(len(t), per, np.int8))
        parts["frame"].append(frames[nr])
        parts["ball_vis"].append(src["ball_vis"][nr])
        parts["ball_raw_vis"].append(src["ball_raw_vis"][nr])
        parts["nvis"].append(src["nvis"][nr])
        parts["video"].append((src["nraw"][nr] > 0) | (src["ball_raw_vis"][nr] > 0))
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    used = np.isfinite(cat["players"][..., 0]).any(0)       # drop squad members never on the pitch
    cat["players"] = cat["players"][:, used]
    base = dict(players=cat["players"].astype(np.float32), ball=cat["ball"].astype(np.float32),
                ball_pff=cat["ball_pff"].astype(np.float32),
                half=cat["half"], frame=cat["frame"].astype(np.int32),
                team_of_player=src["team_of_player"][used].astype(np.int8),
                ball_vis=cat["ball_vis"], ball_raw_vis=cat["ball_raw_vis"],
                nvis=cat["nvis"].astype(np.int8), video=cat["video"])
    for name, (status, poss) in variants.items():
        d = os.path.join(VAR, name)
        os.makedirs(d, exist_ok=True)
        st, po = status[cat["src"]].astype(np.int8), poss[cat["src"]].astype(np.int8)
        np.savez_compressed(os.path.join(d, "PFF-" + game + ".npz"), status=st, poss=po, **base)
        if name == L.CHOSEN:
            os.makedirs(OUT, exist_ok=True)
            np.savez_compressed(os.path.join(OUT, "PFF-" + game + ".npz"), status=st, poss=po, **base)
    return game, len(cat["half"])


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
