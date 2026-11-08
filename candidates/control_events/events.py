"""Parse the DFL event XMLs (IDSSE, CC BY 4.0) and place every event on the tracking frame rows.

Rows are the rows of $MTAH_WORK/idsse/npz/<match>.npz (= gt in bench.dfl2_prepared): both halves'
ball frames, halves concatenated, 25 fps without gaps. frametimes.py stores each row's DFL frame
number N and UTC time T; an event's EventTime (local time with offset) becomes the row whose T is
nearest, inside the half that contains it.

Each event is a dict:
    id, t (epoch s), row (int, -1 outside both halves), calc_row (row of CalculatedFrame or -1),
    kind (the top-level child tag), tags [(tag, attrs)] (all nested elements in order),
    team (0 home / 1 away / -1), and the derived control fields of control_actions().
Delete events (empty placeholders of withdrawn events) are dropped.
"""

import os
import re
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get("MTAH_WORK", os.path.join(ROOT, "external", "work"))
EV_DIR = os.path.join(WORK, "idsse", "events")
RAW_DIR = os.path.join(WORK, "idsse", "raw")
NPZ_DIR = os.path.join(WORK, "idsse", "npz")

RE_ATTR = re.compile(r'([\w-]+)="([^"]*)"')
RE_EVENT = re.compile(r'<Event\b([^>]*)>(.*?)</Event>', re.S)
RE_TAG = re.compile(r'<(\w+)\b([^>]*?)/?>')

SET_PIECES = ("KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty")


def _file(match, prefix, folder):
    for f in os.listdir(folder):
        if f.startswith(prefix) and match in f:
            return os.path.join(folder, f)
    raise FileNotFoundError(prefix + match)


def teams(match):
    """{DFL team id: 0 home / 1 away} from the matchinformation XML."""
    text = open(_file(match, "DFL_02_01_matchinformation", RAW_DIR), encoding="utf-8").read()
    out = {}
    for m in re.finditer(r'<Team\b([^>]*)>', text):
        a = dict(RE_ATTR.findall(m.group(1)))
        if "TeamId" in a and "Role" in a:
            out[a["TeamId"]] = 0 if a["Role"] == "home" else 1
    return out


def frametimes(match):
    z = np.load(os.path.join(EV_DIR, match + "_frametimes.npz"))
    return z["N"], z["t"]


def _epoch(s):
    return datetime.fromisoformat(s).timestamp()


def parse(match):
    """All non-deleted events of a match, sorted by time, with rows."""
    tm = teams(match)
    N, t = frametimes(match)
    # half boundaries in rows: the one jump in t
    cut = int(np.argmax(np.diff(t))) + 1
    halves = [(0, cut), (cut, len(t))]
    nrow = {int(n): i for i, n in enumerate(N)}
    text = open(_file(match, "DFL_03_02_events_raw", EV_DIR), encoding="utf-8").read()
    out = []
    for m in RE_EVENT.finditer(text):
        ea = dict(RE_ATTR.findall(m.group(1)))
        tags = [(g.group(1), dict(RE_ATTR.findall(g.group(2)))) for g in RE_TAG.finditer(m.group(2))]
        if not tags or tags[0][0] == "Delete":
            continue
        te = _epoch(ea["EventTime"])
        row = -1
        for a, b in halves:
            if t[a] - 1.0 <= te <= t[b - 1] + 1.0:
                row = int(np.clip(a + round((te - t[a]) / 0.04), a, b - 1))
        calc = nrow.get(int(ea["CalculatedFrame"]), -1) if "CalculatedFrame" in ea else -1
        ev = dict(id=ea.get("EventId"), t=te, row=row, calc_row=calc, kind=tags[0][0], tags=tags,
                  x=float(ea["X-Position"]) if "X-Position" in ea else np.nan,
                  y=float(ea["Y-Position"]) if "Y-Position" in ea else np.nan)
        ev.update(control_actions(tags, tm))
        out.append(ev)
    out.sort(key=lambda e: e["t"])
    return out


def control_actions(tags, tm):
    """Derived fields.

    team      team of the acting player of the event's main action (-1 if none)
    ctrl      list of (team, label) controlled actions this event shows (see README, rule C1)
    touch     list of (team, label) uncontrolled ball contacts (kept for the report only)
    restart   team taking a set piece, else -1 (a KickOff takes its nested Play's team)
    player    DFL person id of the acting player (Player, or the tackle Winner), else None
    """
    team, ctrl, touch, restart, player = -1, [], [], -1, None

    def T(team_id):
        return tm.get(team_id, -1)

    for tag, a in tags:
        if tag in SET_PIECES:
            restart = T(a.get("Team")) if "Team" in a else -2
        if player is None and tag != "Foul":
            player = a.get("Player") or a.get("Winner")
        if tag == "Play" and restart == -2:
            restart = T(a.get("Team"))
        if tag == "Play":                      # pass or cross (successful or not)
            ctrl.append((T(a.get("Team")), "cross" if any(g == "Cross" for g, _ in tags) else "pass"))
            team = T(a.get("Team")) if team < 0 else team
        elif tag == "ShotAtGoal":
            ctrl.append((T(a.get("Team")), "shot"))
            team = T(a.get("Team")) if team < 0 else team
        elif tag == "BallClaiming":            # InterceptedBall, BallClaimed, BallHeld (keeper)
            ctrl.append((T(a.get("Team")), "claim_" + a.get("Type", "")))
            team = T(a.get("Team"))
        elif tag == "TacklingGame":
            w, role, res = T(a.get("WinnerTeam")), a.get("WinnerRole"), a.get("WinnerResult")
            lo, lrole = T(a.get("LoserTeam")), a.get("LoserRole")
            if role == "withoutBallControl" and a.get("PossessionChange") == "true":
                ctrl.append((w, "tackle_won"))          # winner took the ball off the holder
            elif role == "withBallControl" and res in ("dribbledAround", "ballControlRetained"):
                ctrl.append((w, "dribble"))             # holder kept control through the duel
            elif role == "withBallControl":
                ctrl.append((w, "duel_holder"))         # DFL: winner had the ball under control
            else:
                touch.append((w, "duel_" + str(res)))
            if lrole == "withBallControl" and lo >= 0:
                ctrl.append((lo, "duel_holder"))        # DFL: loser had the ball under control
            team = w
        elif tag in ("Run", "Nutmeg"):
            ctrl.append((T(a.get("Team")), "dribble"))
            team = T(a.get("Team")) if team < 0 else team
        elif tag == "OtherBallAction":
            touch.append((T(a.get("Team")), "clearance" if a.get("DefensiveClearance") == "true"
                          else "other_ball_action"))
            team = T(a.get("Team"))
        elif tag == "BallDeflection":
            touch.append((T(a.get("Team")), "deflection"))
    ctrl = [c for c in ctrl if c[0] >= 0]
    return dict(team=team, ctrl=ctrl, touch=touch, restart=restart, player=player)


def persons(match):
    """npz player columns: sorted PersonIds of tracked team members (as fetch_idsse orders them)."""
    tm = teams(match)
    ids = []
    for line in open(os.path.join(EV_DIR, match + "_persons.txt")):
        a = dict(RE_ATTR.findall(line))
        if a.get("TeamId") in tm:
            ids.append(a["PersonId"])
    return {p: i for i, p in enumerate(sorted(set(ids)))}


def restarts(status, half):
    """Rows where play (re)starts: BallStatus 0 -> 1, plus each half's first alive row."""
    on = set((np.where((status[1:] == 1) & (status[:-1] != 1))[0] + 1).tolist())
    for h in np.unique(half):
        idx = np.where((half == h) & (status == 1))[0]
        if len(idx):
            on.add(int(idx[0]))
    return np.array(sorted(on))


GEOM_KINDS = ("Play", "ShotAtGoal", "BallClaiming", "OtherBallAction", "TacklingGame")


def _player_events(evs, pid):
    return [e for e in evs if e["row"] >= 0 and e["player"] in pid and e["restart"] < 0
            and e["kind"] in GEOM_KINDS]


def geom_offset(evs, z, pid, near=1.5, search=4.0, fps=25):
    """Per-game clock offset (rows) of the event feed against the POSITIONS: the shift that
    maximises the share of open-play player events whose acting player is within `near` m of the
    ball. Uses positions only, no BallStatus / BallPossession flag."""
    P, B = z["players"], z["ball"]
    n = len(B)
    es = _player_events(evs, pid)
    rows = np.array([e["row"] for e in es])
    cols = np.array([pid[e["player"]] for e in es])
    lags = np.arange(-int(search * fps), int(search * fps) + 1)
    share = []
    for L in lags:
        r = np.clip(rows + L, 0, n - 1)
        d = np.linalg.norm(P[r, cols] - B[r], axis=1)
        share.append(np.nanmean(d < near))
    share = np.convolve(share, np.ones(5) / 5, mode="same")     # 0.2 s smoothing of the profile
    return int(lags[np.argmax(share)])


def align(match, evs, z, window=1.5, max_dist=2.0, fps=25):
    """Set e["frame"], the event's row on the positions' clock.

    0. an event with a CalculatedFrame (DFL's own tracking-based frame, given for shots and a few
       others) takes that frame: half the shots' EventTimes are logged 2-10 s late;
    1. shift every other event by geom_offset (one constant per game);
    2. an event with a tracked acting player snaps to the row within +-window s of the shifted
       row where that player is closest to the ball, if that distance is <= max_dist m; other
       events keep the shifted row.
    Returns (offset s, share of player events snapped).
    """
    ball, players = z["ball"], z["players"]
    n = len(ball)
    pid = persons(match)
    off = geom_offset(evs, z, pid)
    w = int(round(window * fps))
    snapped = tried = 0
    for e in evs:
        if e["row"] < 0:
            e["frame"] = -1
            continue
        r = int(np.clip(e["row"] + off, 0, n - 1))
        e["frame"] = r
        e["snapped"] = False
        if e["calc_row"] >= 0:
            e["frame"] = e["calc_row"]
            e["snapped"] = True
        elif e["player"] in pid:
            tried += 1
            a, b = max(0, r - w), min(n, r + w + 1)
            d = np.linalg.norm(players[a:b, pid[e["player"]]] - ball[a:b], axis=1)
            d = np.where(np.isnan(d), np.inf, d)
            j = int(np.argmin(d))
            if d[j] <= max_dist:
                e["frame"] = a + j
                e["snapped"] = True
                snapped += 1
    return off / fps, snapped / max(tried, 1)


def flag_lag(evs, z, lo=-1.0, hi=2.0, fps=25):
    """Per-game lag (rows) of the operator-set DFL flags behind the positions' clock: the shift in
    [lo, hi] s that maximises the share of aligned open-play controlled actions whose team the
    BallPossession flag names. One constant per game, so it cannot move single events into
    single spells; README "Alignment" explains why it is needed (the flag is set live by hand)."""
    status, poss = z["status"], z["poss"]
    n = len(status)
    es = [e for e in evs if e["frame"] >= 0 and e["ctrl"] and e["restart"] < 0]
    rows = np.array([e["frame"] for e in es])
    tms = np.array([e["ctrl"][0][0] for e in es])
    lags = np.arange(int(lo * fps), int(hi * fps) + 1)
    acc = []
    for L in lags:
        r = np.clip(rows + L, 0, n - 1)
        ok = status[r] == 1
        acc.append(np.mean((poss[r][ok] - 1) == tms[ok]))
    acc = np.convolve(acc, np.ones(5) / 5, mode="same")
    return int(lags[np.argmax(acc)])


def load_aligned(match):
    """(events with "frame" (positions clock) and "fframe" (flag clock), npz, info dict)."""
    z = np.load(os.path.join(NPZ_DIR, match + ".npz"))
    evs = parse(match)
    off, snap = align(match, evs, z)
    lag = flag_lag(evs, z)
    n = len(z["status"])
    for e in evs:
        e["fframe"] = int(min(e["frame"] + lag, n - 1)) if e["frame"] >= 0 else -1
    return evs, z, dict(offset_s=off, snapped=snap, flag_lag_s=lag / 25.0)
