"""Frame labels (status, possessing team) for the PFF FC World Cup 2022 games, from PFF's own events.

No hand labelling: every label is a deterministic function of PFF's game events (attached to the
tracking frames: type, start_frame, home_team, home_ball, sequence) and of the event-data file
(set-piece type, out type, possession-event types per game event). Everything is computed at the
source rate (29.97 fps, PFF frameNum); convert.py samples it to 25 fps.

Ball status (identical in every variant)
    dead (0) from each OUT game event's start_frame (PFF: "ball out of play"; outType T = out of
    touch / over the goal line, W = whistle, which covers fouls, offsides and other stoppages,
    H / A = goal) until the start_frame of the next on-the-ball event of the same period (the
    restart: in all 64 games every set piece except kick-offs follows an OUT, checked by
    check_restarts), or to the end of the period if none follows. Kick-offs start each period; the
    graded rows start at the kick-off, so the pre-kick-off dead time does not exist here. PFF's
    tracking ends each period at its END event.

Possessing team (alive frames), three readings
    touch  the team of the latest on-the-ball game event (OTB / kick-off), i.e. of the player PFF
           charts as on the ball, whatever he does with it (pass, challenge, clearance, rebound,
           block). The analogue of the raw DFL BallPossession flag, which flips on every touch.
    nore   as touch, but game events whose only possession events are rebounds (RE: "a touch where
           a player has no intention to control or distribute": parries, blocks, inadvertent
           touches) do not change the team. The narrowest reading of CHALLENGE.md's "a deflection
           of a pass that does not intercept it" / "a simple touch without control".
    seq    PFF's own possession: home_ball of the latest on-the-ball game event. PFF groups events
           into sequences ("a sequence is represented by an uninterrupted possession by one
           team", event spec 2.5); clearances, rebounds, blocks and lost challenges by the
           defending team stay inside the attacking team's sequence until that team (or the
           other) gains control. This is the "controlled possession" CHALLENGE.md asks for.
A team change takes effect at the event's start_frame (the first touch).

Chosen: touch (label_stats.py, 64 games, 2026-11-08). Mean over games of the length fingerprint
(spellF1 of a perfect submission 3 / 8 / 25 frames late) and possessions per 90 graded minutes:

    full-match truth (EVALUATION.md)   0.982 / 0.893 / 0.654   ~306 / 90 (17 in the public 5 min)
    PFF touch                          0.977 / 0.902 / 0.632    314 / 90
    PFF nore                           0.994 / 0.939 / 0.682    284 / 90
    PFF seq                            0.994 / 0.954 / 0.745    246 / 90
    DFL raw flag                       0.959 / 0.862 / 0.557    412 / 90
Dead share 0.411 in every variant.

Why touch although seq is the closest to CHALLENGE.md's prose ("controlled" possession): the same
document states how the truth is actually made -- mechanically, from the corrected tracking's
"team the tracking names as owning" the ball, with nothing smoothed -- and says the ground truth
wins over any reading of the prose. touch is exactly that kind of signal (the team of the player
the data provider puts on the ball), and both the fingerprint and the possession rate agree with
the truth's published numbers, whereas seq is an analyst-defined possession with 20% fewer and
longer spells (0.954 / 0.745 is what a 1 to 2 s fold of the DFL flag gives, which the round-2
analysis rejected). seq and nore are kept as reference readings, and the bench also folds touch at
0.5 / 1 / 1.5 / 2 s.
"""

import json
import os

import numpy as np

CHOSEN = "touch"
VARIANTS = ("touch", "nore", "seq")
ON_BALL = ("OTB", "FIRSTKICKOFF", "SECONDKICKOFF")
KICKOFF = {"FIRSTKICKOFF": 1, "SECONDKICKOFF": 2}


def event_json(game, raw_dir):
    with open(os.path.join(raw_dir, "Event Data", game + ".json")) as h:
        rows = json.load(h)
    out = {}
    for r in rows:
        g = r.get("gameEvents") or {}
        e = out.setdefault(int(r["gameEventId"]), dict(setpiece=g.get("setpieceType"),
                                                         out=g.get("outType"),
                                                         video_missing=g.get("videoMissing"),
                                                         ptypes=[], fouls=[]))
        pe = r.get("possessionEvents") or {}
        if pe.get("possessionEventType"):
            e["ptypes"].append(pe["possessionEventType"])
        f = r.get("fouls") or {}
        if f.get("foulType"):
            e["fouls"].append(f["foulType"])
    return out


def kickoffs(ev):
    k = {}
    for e in ev["game_events"]:
        if e["type"] in KICKOFF:
            k.setdefault(KICKOFF[e["type"]], int(e["start_frame"]))
    return k


def _period_events(ev, period, first, last):
    """Game events of one period (by start_frame inside [first, last]), sorted."""
    out = [e for e in ev["game_events"] if e["start_frame"] is not None
           and first <= int(e["start_frame"]) <= last and e["type"] != "END"]
    out.sort(key=lambda e: (int(e["start_frame"]), e["id"]))
    return out


def build(src, ev, ej, variant):
    """(status, poss) per source row; poss 1 home, 2 away."""
    frames, period = src["frame"], src["period"]
    status = np.ones(len(frames), np.int8)
    poss = np.zeros(len(frames), np.int8)
    for per in (1, 2):
        sel = np.where(period == per)[0]
        f = frames[sel]
        evs = _period_events(ev, per, int(f[0]), int(f[-1]))
        onball = [e for e in evs if e["type"] in ON_BALL]
        # status
        for i, e in enumerate(evs):
            if e["type"] != "OUT":
                continue
            a = int(e["start_frame"])
            nxt = [o for o in onball if int(o["start_frame"]) > a]
            b = int(nxt[0]["start_frame"]) if nxt else int(f[-1]) + 1
            status[sel[(f >= a) & (f < b)]] = 0
        # team
        changes = []
        for e in onball:
            info = ej.get(e["id"], {})
            if variant == "seq":
                hb = e.get("home_ball")
                if hb is None:
                    continue
                team = 1 if hb else 2
            else:
                ht = e.get("home_team")
                if ht is None:
                    continue
                if variant == "nore":
                    pt = set(info.get("ptypes", [])) - {"IT"}
                    if pt and pt <= {"RE"}:
                        continue
                team = 1 if int(ht) == 1 else 2
            changes.append((int(e["start_frame"]), team))
        cur = changes[0][1] if changes else 1
        out = np.full(len(f), cur, np.int8)
        for a, team in changes:
            out[f >= a] = team
        poss[sel] = out
    return status, poss


def build_all(game, src, ev, fps_src, raw_dir=None):
    raw_dir = raw_dir or os.path.join(os.environ.get("MTAH_WORK", ""), "pff_wc2022", "raw")
    ej = event_json(game, raw_dir)
    return {v: build(src, ev, ej, v) for v in VARIANTS}


def check_restarts(ev, ej):
    """Counts of on-ball set pieces (setpieceType not O) by whether an OUT precedes them since the
    previous on-ball event; kick-offs at period starts are counted separately."""
    from collections import Counter
    c = Counter()
    prev = None
    for e in sorted((e for e in ev["game_events"] if e["start_frame"] is not None),
                    key=lambda e: (int(e["start_frame"]), e["id"])):
        if e["type"] in ON_BALL:
            sp = ej.get(e["id"], {}).get("setpiece")
            if e["type"] in KICKOFF:
                c["period_kickoff"] += 1
            elif sp not in ("O", None):
                c[("after_out" if prev == "OUT" else "no_out", sp)] += 1
            prev = "OTB"
        elif e["type"] == "OUT":
            prev = "OUT"
    return c
