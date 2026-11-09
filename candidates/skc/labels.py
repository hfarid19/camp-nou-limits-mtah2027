"""Frame labels (status, possessing team) for the SkillCorner A-League 2024/25 open games, by script.

No hand labelling: every label is a deterministic function of SkillCorner's own files for the game:
the dynamic-events CSV (player_possession rows: frame_start / frame_end, team, start_type, end_type,
game_interruption_before / _after; on_ball_engagement rows: frame_end, team, end_type), the
phases-of-play CSV (frame_start / frame_end, team_in_possession; "phases of play are only defined
when the ball is in play") and the tracking file's per-frame `possession.group`. Everything is
computed at the source rate (10 fps, SkillCorner frame index); convert.py samples it to 25 fps.
Graded rows are each period's [start_frame, end_frame] from the match JSON.

Ball status, two readings
    phase  alive inside a phase of play [frame_start, frame_end), dead in every gap between phases
           (all gaps are >= 1 s; 5th percentile 5 s) and before the first / after the last phase.
    event  dead from the frame_end of a player possession or on-ball engagement whose
           game_interruption_after is set (throw_in / free_kick / corner / goal_kick / goal, for or
           against), or whose end_type is foul_suffered / foul_committed, until the frame_start of
           the next player possession whose game_interruption_before is set or whose start_type is a
           restart (throw_in_ / free_kick_ / corner_ / goal_kick_ reception or interception).
           SkillCorner logs the restart's receiver, not its taker, so the dead run also covers the
           restart pass's flight; the ball's exit is also later than the last possession's end.

Possessing team (alive frames), four readings
    touch   the team of the latest player_possession (from its frame_start): the player the
            tracking puts on the ball. Includes one-frame possessions (first-time passes,
            clearances, interceptions). The analogue of PFF "touch" and of the DFL flag.
    touch_eng  touch, plus on-ball engagements whose end_type is direct_disruption or direct_regain
            (the engaging defender plays the ball): the team flips to the engager's team at the
            engagement's frame_end, until the next player possession starts. Adds the short
            changes that touch misses (tackles and pokes without a following possession).
    phase   the phase of play's team_in_possession: SkillCorner's team possession, which groups a
            team's possessions into one phase and ignores loose touches (the analogue of PFF seq).
    flag    the tracking's per-frame possession.group, held through frames where it is empty.
A team change takes effect at the event's frame. Before a period's first possession, the period's
first possessing team. The dead mask (status) is applied on top, so a team reading only matters on
alive frames.

Chosen: touch_eng (fingerprint 0.980 / 0.897 / 0.663, 279 / 90, closest to the truth 0.982 / 0.893 / 0.654);
phase is the "controlled possession" reference reading. Table: $MTAH_WORK/skc_aleague/README.md (label_stats.py).
"""

import json
import os

import numpy as np
import pandas as pd

CHOSEN = "touch_eng"
STATUS_CHOSEN = "phase"
TEAM_VARIANTS = ("touch", "touch_eng", "phase", "flag")
# variant name -> (status reading, team reading)
VARIANTS = {"touch": ("phase", "touch"), "touch_eng": ("phase", "touch_eng"), "phase": ("phase", "phase"),
            "flag": ("phase", "flag"), "touch_evst": ("event", "touch")}
RESTART_START = ("throw_in_reception", "throw_in_interception", "free_kick_reception", "free_kick_interception",
                 "corner_reception", "corner_interception", "goal_kick_reception", "goal_kick_interception")
FOUL_END = ("foul_suffered", "foul_committed")
DIRECT = ("direct_disruption", "direct_regain")


def read_game(raw_dir, game):
    d = os.path.join(raw_dir, "data", "matches", game)
    with open(os.path.join(d, game + "_match.json")) as h:
        meta = json.load(h)
    ev = pd.read_csv(os.path.join(d, game + "_dynamic_events.csv"), low_memory=False)
    ph = pd.read_csv(os.path.join(d, game + "_phases_of_play.csv"), low_memory=False)
    return meta, ev, ph


def periods(meta, n=None):
    """{period: (first, last)} inclusive source frames, clipped to the tracking file's n rows."""
    hi = (n - 1) if n else 10 ** 9
    return {int(p["period"]): (int(p["start_frame"]), min(int(p["end_frame"]), hi)) for p in meta["match_periods"]
            if int(p["period"]) in (1, 2)}


def side_of(meta):
    home, away = int(meta["home_team"]["id"]), int(meta["away_team"]["id"])
    return lambda t: 1 if int(t) == home else 2 if int(t) == away else 0


def status_phase(n, ph):
    st = np.zeros(n, np.int8)
    for a, b in zip(ph.frame_start.astype(int), ph.frame_end.astype(int)):
        st[a:b] = 1
    return st


def status_event(n, ev, pers):
    st = np.zeros(n, np.int8)
    pp = ev[ev.event_type == "player_possession"]
    for _, (a0, b0) in pers.items():
        st[a0:b0 + 1] = 1
    stop = ev[(ev.event_type.isin(["player_possession", "on_ball_engagement"]))
              & (ev.game_interruption_after.notna() | ev.end_type.isin(FOUL_END))]
    rest = pp[pp.game_interruption_before.notna() | pp.start_type.isin(RESTART_START)]
    rs = np.sort(rest.frame_start.astype(int).values)
    for a in np.sort(stop.frame_end.astype(int).values):
        per = [b0 for a0, b0 in pers.values() if a0 <= a <= b0]
        if not per:
            continue
        j = np.searchsorted(rs, a, side="right")
        b = min(rs[j], per[0] + 1) if j < len(rs) else per[0] + 1
        st[a:b] = 0
    # before the first possession of each period: dead (kick-off taker is not a possession)
    for a0, b0 in pers.values():
        s = pp[(pp.frame_start >= a0) & (pp.frame_start <= b0)].frame_start
        if len(s):
            st[a0:int(s.min())] = 0
    return st


def _hold(changes, a0, b0, out):
    """Write a piecewise-constant team from (frame, team) changes into out[a0:b0+1]."""
    _hold_ordered(sorted(changes), a0, b0, out)


def team_reading(n, ev, ph, flag, pers, side, variant):
    out = np.zeros(n, np.int8)
    pp = ev[ev.event_type == "player_possession"]
    for _, (a0, b0) in pers.items():
        if variant in ("touch", "touch_eng"):
            ch = [(int(f), 1, side(t)) for f, t in zip(pp.frame_start, pp.team_id)]
            if variant == "touch_eng":
                ob = ev[(ev.event_type == "on_ball_engagement") & ev.end_type.isin(DIRECT)]
                ch += [(int(f), 0, side(t)) for f, t in zip(ob.frame_end, ob.team_id)]
            # on one frame a possession start (priority 1) wins over an engagement's end (0)
            _hold_ordered([(f, t) for f, _, t in sorted(ch)], a0, b0, out)
            continue
        elif variant == "phase":
            ch = [(int(f), side(t)) for f, t in zip(ph.frame_start, ph.team_in_possession_id)]
        elif variant == "flag":
            idx = np.where(flag[a0:b0 + 1] > 0)[0] + a0
            keep = np.r_[True, flag[idx][1:] != flag[idx][:-1]] if len(idx) else np.zeros(0, bool)
            ch = [(int(f), int(flag[f])) for f in idx[keep]]
        else:
            raise ValueError(variant)
        _hold(ch, a0, b0, out)
    return out


def _hold_ordered(seq, a0, b0, out):
    """As _hold, but keeps the given order for changes on the same frame (the last one wins)."""
    seq = [(f, t) for f, t in seq if a0 <= f <= b0 and t > 0]
    if not seq:
        return
    out[a0:b0 + 1] = seq[0][1]
    for f, t in seq:
        out[f:b0 + 1] = t


def build_all(raw_dir, game, n, flag):
    """{variant: (status, poss)} at the source rate; poss 1 home, 2 away (0 outside the periods)."""
    meta, ev, ph = read_game(raw_dir, game)
    pers, side = periods(meta, n), side_of(meta)
    stats = {"phase": status_phase(n, ph), "event": status_event(n, ev, pers)}
    teams = {v: team_reading(n, ev, ph, flag, pers, side, v) for v in TEAM_VARIANTS}
    graded = np.zeros(n, bool)
    for a0, b0 in pers.values():
        graded[a0:b0 + 1] = True
    out = {}
    for name, (s, t) in VARIANTS.items():
        st = np.where(graded, stats[s], 0).astype(np.int8)
        po = np.where(graded, teams[t], 0).astype(np.int8)
        out[name] = (st, po)
    return out
