#!/usr/bin/env python3
"""SoccerTrack v2 (10 university matches, CC BY 4.0) -> the IDSSE npz format the DFL benches read.

    python -I candidates/soccertrack/convert.py parse  [games...]   # provider XML -> per-half cache
    python -I candidates/soccertrack/convert.py write  [games...]   # cache + labels -> npz
    python -I candidates/soccertrack/convert.py all    [games...]
    (--jobs 2 runs games in parallel; no games = all 10)

Input (never committed, read as data only): $MTAH_WORK/soccertrack/raw = `hf download
atomscott/soccertrack-v2 --include raw/* bas/* ball/*` (release v1.2). Only raw/ (the provider's
tracking and event exports, calibration), bas/ (ball-action events) and ball/ (ball arrays) are read;
gsr/ and videos/ are not downloaded. Output:

    $MTAH_WORK/soccertrack/src/<game>_<half>.npz                 per-half cache, video-frame clock
    $MTAH_WORK/soccertrack/npz/ST-<game>.npz                     chosen labels (labels.CHOSEN)
    $MTAH_WORK/soccertrack/variants/<variant>/ST-<game>.npz      every label variant

Keys as the IDSSE npz (players (N, P, 2) corner-origin metres on 105 x 68, NaN when the person is not
tracked; ball (N, 2) never NaN; status 1 alive / 0 dead; poss 1 home / 2 away; half 1 / 2; frame =
1-based video frame of the half; team_of_player 0 home / 1 away), plus player_id (P,), flag (N,) int8
the provider's ballStatus (0 BALLOUT, 1 HOME, 2 AWAY, 3 NEUTRAL, -1 none) and offset (2,) the
provider-row offset of each half.

Native 25 fps, no resampling. Clock: the provider's tracking (frameNumber / matchTime) is put on the
video-frame clock of ball/ and bas/ by the per-half row offset at which the provider's ball equals the
ball/ array (exact to 1e-6 m at the right offset; this also carries release v1.2's re-synchronisation
of match 132877). Provider events (player_nodes.csv, match clock in ms) go to the tracking row with the
nearest matchTime. Rows: each half's graded block (labels.py: first to last non-BALLOUT frame).

Positions: the provider's tracking, loc in [0, 1] of the 105 x 68 pitch -> (x * 105, y * 68). Short
holes in a player's track (<= 2 s) are linearly interpolated; longer absences stay NaN.

THE BALL IS NOT MEASURED. The provider's ball (= ball/) is a reconstruction anchored at the annotated
events and moved in straight constant-speed lines between them (dataset README), i.e. the synthetic
ball of candidates/_scratch/ingest_synthball.py (Task 2) with denser anchors. Holes are filled by
linear interpolation in time.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labels as L    # noqa: E402

WORK = os.environ.get("MTAH_WORK") or os.path.join(os.path.dirname(os.path.dirname(HERE)), "external", "work")
BASE = os.path.join(WORK, "soccertrack")
RAW = os.path.join(BASE, "raw")
SRC = os.path.join(BASE, "src")
OUT = os.path.join(BASE, "npz")
VAR = os.path.join(BASE, "variants")
PITCH = (105.0, 68.0)
HALVES = (("FIRST_HALF", "1st", 1), ("SECOND_HALF", "2nd", 2))
CODE = {"BALLOUT": 0, "HOME": 1, "AWAY": 2, "NEUTRAL": 3}
RF = re.compile(r'<frame matchTime="([-\d.]+)" frameNumber="(\d+)" eventPeriod="(\w+)" ballStatus="(\w*)"')
RP = re.compile(r'<(?:player|ball) playerId="(\w+)" loc="\[([-\d.e]+|nan), ([-\d.e]+|nan)\]"')
RA = re.compile(r'(\w+)="([^"]*)"')
MAX_HOLE = 50


def games():
    return sorted(g for g in os.listdir(os.path.join(RAW, "raw")) if g.isdigit())


def metadata(game):
    text = open(os.path.join(RAW, "raw", game, game + "_tracker_box_metadata.xml"), encoding="utf-8").read()
    side = {}
    for m in re.finditer(r"<team ([^>]*)/>", text):
        a = dict(RA.findall(m.group(1)))
        side[a["id"]] = 0 if a["side"] == "home" else 1
    team = {}
    for m in re.finditer(r"<player ([^>]*)/>", text):
        a = dict(RA.findall(m.group(1)))
        pid = a.get("id") or a.get("playerId")          # roster entries and per-period lineup entries
        if pid and a.get("teamId") in side:
            team[pid] = side[a["teamId"]]
    return team, side


def read_tracking(game):
    path = os.path.join(RAW, "raw", game, game + "_tracker_box_data.xml")
    out = {p: dict(mt=[], fn=[], st=[], pl=[], ball=[]) for p, _, _ in HALVES}
    cur = None
    with open(path, encoding="utf-8") as h:
        for line in h:
            m = RF.search(line)
            if m:
                cur = out.get(m.group(3))
                if cur is not None:
                    cur["mt"].append(float(m.group(1)))
                    cur["fn"].append(int(m.group(2)))
                    cur["st"].append(CODE.get(m.group(4), -1))
                    cur["pl"].append({})
                    cur["ball"].append((np.nan, np.nan))
                continue
            if cur is None:
                continue
            m = RP.search(line)
            if m:
                xy = (float(m.group(2)), float(m.group(3)))
                if m.group(1) == "ball":
                    cur["ball"][-1] = xy
                else:
                    cur["pl"][-1][m.group(1)] = xy
    return out


def find_offset(pb, z):
    """Row offset off (provider row = ball/ row + off) and y sign at which the provider ball equals ball/."""
    bx, by = (pb[:, 0] - 0.5) * PITCH[0], (pb[:, 1] - 0.5) * PITCH[1]
    n = len(z["x"])
    a = np.arange(0, n, 7)
    best = (np.inf, 0, 1)
    for off in range(-300, 301):
        b = a + off
        ok = (b >= 0) & (b < len(bx))
        if ok.sum() < 100:
            continue
        for sg in (1, -1):
            d = np.nanmean(np.hypot(bx[b[ok]] - z["x"][a[ok]], sg * by[b[ok]] - z["y"][a[ok]]))
            if d < best[0]:
                best = (d, off, sg)
    return best


def parse(game):
    team, _ = metadata(game)
    tr = read_tracking(game)
    os.makedirs(SRC, exist_ok=True)
    info = []
    for pname, hname, hi in HALVES:
        t = tr[pname]
        z = np.load(os.path.join(RAW, "ball", "{}_{}_ball.npz".format(game, hname)), allow_pickle=False)
        pb = np.array(t["ball"], np.float64)
        dist, off, sg = find_offset(pb, z)
        assert dist < 0.01, (game, hname, dist)
        n = len(z["x"])
        pids = sorted({p for d in t["pl"] for p in d if p in team})
        col = {p: j for j, p in enumerate(pids)}
        players = np.full((n, len(pids), 2), np.nan, np.float32)
        flag = np.full(n, -1, np.int8)
        ball = np.full((n, 2), np.nan, np.float32)
        mt = np.full(n, np.nan)
        for a in range(n):
            b = a + off
            if 0 <= b < len(t["pl"]):
                for p, xy in t["pl"][b].items():
                    if p in col:
                        players[a, col[p]] = (xy[0] * PITCH[0], xy[1] * PITCH[1])
                flag[a] = t["st"][b]
                ball[a] = (pb[b, 0] * PITCH[0], pb[b, 1] * PITCH[1])
                mt[a] = t["mt"][b]
        zs = z["status"].astype(np.int8)
        agree = float(np.mean(zs[zs >= 0] == flag[zs >= 0]))
        np.savez_compressed(os.path.join(SRC, "{}_{}.npz".format(game, hname)), players=players, flag=flag,
                            ball=ball, match_time=mt, player_id=np.array([int(p) for p in pids], np.int64),
                            team_of_player=np.array([team[p] for p in pids], np.int8), offset=off, ysign=sg)
        info.append((hname, n, off, sg, round(agree, 5)))
    return game, info


def events(game, hname, hi, src, team):
    """BAS events and provider events of one half as (row, team 1|2, label) lists."""
    with open(os.path.join(RAW, "bas", game, game + "_12_class_events.json")) as h:
        acts = json.load(h)["actions"]
    n = len(src["flag"])
    ids = {str(p): int(t) for p, t in zip(src["player_id"], src["team_of_player"])}
    lr = {}
    for a in acts:
        if a["gameTime"].startswith(str(hi)) and str(a.get("player_id")) in team:
            lr.setdefault(a["team"], []).append(team[str(a["player_id"])])
    lr = {k: int(round(np.mean(v))) for k, v in lr.items()}
    bas = []
    for a in acts:
        if not a["gameTime"].startswith(str(hi)):
            continue
        r = int(a["frame"]) - 1
        pid = str(a.get("player_id") or "")
        tm = team.get(pid, lr.get(a.get("team")))
        if 0 <= r < n and tm is not None:
            bas.append((r, tm + 1, a["label"]))
    import csv
    nodes = []
    mt = src["match_time"]
    ok = np.isfinite(mt)
    rows_ok, mt_ok = np.where(ok)[0], mt[ok]
    with open(os.path.join(RAW, "raw", game, game + "_player_nodes.csv"), encoding="utf-8") as h:
        for rec in csv.DictReader(h):
            if rec["event_period"] != dict(HALVES_BY_I)[hi] or not rec["player_id"]:
                continue
            pid = str(int(float(rec["player_id"])))
            if pid not in team:
                continue
            t = float(rec["event_time"])
            j = int(np.clip(np.searchsorted(mt_ok, t), 1, len(mt_ok) - 1))
            j = j - 1 if abs(mt_ok[j - 1] - t) <= abs(mt_ok[j] - t) else j
            nodes.append((int(rows_ok[j]), team[pid] + 1, rec["filtered_event_types"] or ""))
    return bas, nodes, ids


HALVES_BY_I = [(hi, p) for p, _, hi in HALVES]


def fill_holes(p, max_hole=MAX_HOLE):
    out = p.copy()
    for j in range(p.shape[1]):
        ok = np.isfinite(p[:, j, 0])
        if ok.sum() < 2:
            continue
        k = np.where(ok)[0]
        gaps = np.where(np.diff(k) > 1)[0]
        for g in gaps:
            a, b = k[g], k[g + 1]
            if b - a - 1 <= max_hole:
                w = (np.arange(a + 1, b) - a) / (b - a)
                out[a + 1:b, j] = (1 - w)[:, None] * p[a, j] + w[:, None] * p[b, j]
    return out


def write(game):
    team, _ = metadata(game)
    parts = {}
    allp = []
    halves = []
    for pname, hname, hi in HALVES:
        z = np.load(os.path.join(SRC, "{}_{}.npz".format(game, hname)))
        src = {k: z[k] for k in z.files}
        flag = src["flag"]
        bas, nodes, _ = events(game, hname, hi, src, team)
        alive = np.where(np.isin(flag, (1, 2, 3)))[0]
        if not len(alive):                      # M9 / M10: ballStatus is INPLAY everywhere
            alive = np.array(sorted(r for r, _, _ in bas))
        a0, b0 = int(alive[0]), int(alive[-1])
        var = L.build(len(flag), flag, (a0, b0), bas, nodes)
        halves.append((hi, src, a0, b0, var))
        allp += [int(p) for p in src["player_id"]]
    pids = sorted(set(allp))
    col = {p: j for j, p in enumerate(pids)}
    tm = {}
    out = {k: [] for k in ("players", "ball", "half", "frame", "flag")}
    vout = {}
    for hi, src, a0, b0, var in halves:
        n = b0 - a0 + 1
        pl = np.full((n, len(pids), 2), np.nan, np.float32)
        hp = fill_holes(src["players"][a0:b0 + 1])
        for j, p in enumerate(src["player_id"]):
            pl[:, col[int(p)]] = hp[:, j]
            tm[int(p)] = int(src["team_of_player"][j])
        b = src["ball"][a0:b0 + 1].astype(np.float64)
        ok = np.isfinite(b[:, 0])
        idx = np.arange(n)
        b = np.stack([np.interp(idx, idx[ok], b[ok, c]) for c in range(2)], 1)
        out["players"].append(pl)
        out["ball"].append(b.astype(np.float32))
        out["half"].append(np.full(n, hi, np.int8))
        out["frame"].append(np.arange(a0, b0 + 1, dtype=np.int32) + 1)
        out["flag"].append(src["flag"][a0:b0 + 1])
        for name, (st, po) in var.items():
            vout.setdefault(name, ([], []))
            vout[name][0].append(st[a0:b0 + 1])
            vout[name][1].append(po[a0:b0 + 1])
    cat = {k: np.concatenate(v) for k, v in out.items()}
    base = dict(players=cat["players"], ball=cat["ball"], half=cat["half"], frame=cat["frame"], flag=cat["flag"],
                team_of_player=np.array([tm[p] for p in pids], np.int8), player_id=np.array(pids, np.int64),
                offset=np.array([int(h[1]["offset"]) for h in halves]))
    for name, (st, po) in vout.items():
        st, po = np.concatenate(st).astype(np.int8), np.concatenate(po).astype(np.int8)
        assert ((po > 0) | (st == 0)).all(), (game, name)
        d = os.path.join(VAR, name)
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "ST-" + game + ".npz"), status=st, poss=po, **base)
        if name == L.CHOSEN:
            os.makedirs(OUT, exist_ok=True)
            np.savez_compressed(os.path.join(OUT, "ST-" + game + ".npz"), status=st, poss=po, **base)
    nvis = np.isfinite(cat["players"][..., 0]).sum(1)
    return game, len(cat["half"]), len(pids), float(np.mean(nvis)), float(np.mean(nvis < 22))


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
