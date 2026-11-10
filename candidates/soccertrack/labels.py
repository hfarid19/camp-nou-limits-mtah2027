"""Frame labels (status, possessing team) for SoccerTrack v2 (10 university matches, CC BY 4.0), by script.

No hand labelling here: every label is a deterministic function of the dataset's own files, all at
25 fps on the video-frame clock of each half (row k-1 = video frame k):

  flag   the provider tracking's per-frame ballStatus (raw/<m>/<m>_tracker_box_data.xml): HOME, AWAY,
         NEUTRAL (loose ball) or BALLOUT. Identical to the `status` column of ball/<m>_<half>_ball.npz
         (0 BALLOUT, 1 HOME, 2 AWAY, 3 NEUTRAL; checked on every half).
  bas    the 12-class ball-action spotting events (bas/<m>/<m>_12_class_events.json): frame, label
         (PASS, DRIVE, HIGH PASS, CROSS, SHOT, HEADER, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE,
         THROW IN, FREE KICK, OUT, GOAL), acting player (-> team through the tracking metadata).
  nodes  the provider's own event export (raw/<m>/<m>_player_nodes.csv): one row per player action
         (pass, reception folded into the receiver's action, possession, intercept, duel, tackle,
         clearance, block, cut-off, throw-in, ...) on the tracking's match clock.

Ball status, two readings
    flag   dead where ballStatus is BALLOUT.
    bas    dead from each OUT or GOAL event's frame to the frame of the next other event (the restart:
           throw-in, free kick, the goal-kick / corner / kick-off pass), in the same half.

Possessing team (alive frames), three readings
    flag        HOME / AWAY of ballStatus; NEUTRAL (a loose ball) keeps the last team (CHALLENGE.md:
                "in phases where the ball is contested or loose, possession stays with the last team").
    bas_touch   team of the latest BAS event other than OUT / GOAL (the player on the ball).
    nodes_touch team of the latest provider event row with a player and an on-ball action type.
A team change takes effect at the event's frame; before a half's first event, the first event's team.

Graded rows: each half from the first to the last frame whose ballStatus is not BALLOUT (the
pre-kick-off and post-whistle dead time is dropped, as the competition's graded blocks do).
Matches 132831 (M9) and 132877 (M10): the provider's ballStatus is INPLAY on every frame (no team, no
dead ball), so only the bas_all / nodes_all variants exist there, graded from the first to the last
BAS event of each half.

Chosen: bas_all (BAS dead ball + BAS touch; all 10 games; 0.980 / 0.932 / 0.725 at 340 / 90). The
provider flag (0.997 / 0.971 / 0.810, 259 / 90, 8 games) is a controlled-possession reading; mixing
the flag's dead ball with BAS teams (bas_touch) makes sub-second spells at every restart, because
the two clocks disagree by a few frames. Table: $MTAH_WORK/soccertrack/README.md (label_stats.py).
"""

import numpy as np

CHOSEN = "bas_all"
VARIANTS = {"flag": ("flag", "flag"), "bas_touch": ("flag", "bas_touch"), "nodes_touch": ("flag", "nodes_touch"),
            "bas_all": ("bas", "bas_touch"), "nodes_all": ("bas", "nodes_touch")}
DEAD_BAS = ("OUT", "GOAL")
NOT_ON_BALL = {"changeIn", "changeOut", "offside", "foulConceded", "goalAgainst", "ballMissed"}


def _hold(changes, a0, b0, out):
    ch = sorted((f, t) for f, t in changes if a0 <= f <= b0 and t > 0)
    if not ch:
        return
    out[a0:b0 + 1] = ch[0][1]
    for f, t in ch:
        out[f:b0 + 1] = t


def team_flag(flag, a0, b0, out):
    """HOME/AWAY from ballStatus codes (1, 2), NEUTRAL (3) and BALLOUT (0) hold the last team."""
    seg = flag[a0:b0 + 1].astype(np.int8)
    t = np.where(np.isin(seg, (1, 2)), seg, 0)
    idx = np.where(t > 0)[0]
    if not len(idx):
        return
    fill = np.maximum.accumulate(np.where(t > 0, np.arange(len(t)), -1))
    res = np.where(fill >= 0, t[np.maximum(fill, 0)], t[idx[0]])
    out[a0:b0 + 1] = res


def build(n, flag, block, bas, nodes):
    """{variant: (status, poss)} for one half. bas / nodes: lists of (row, team 1|2, label/types)."""
    a0, b0 = block
    stats = {"flag": (flag != 0).astype(np.int8)}
    st = np.ones(n, np.int8)
    rows = sorted((r, lab) for r, _, lab in bas if a0 <= r <= b0)
    for i, (r, lab) in enumerate(rows):
        if lab in DEAD_BAS:
            nxt = [q for q, l2 in rows[i + 1:] if l2 not in DEAD_BAS and q > r]
            st[r:(nxt[0] if nxt else b0 + 1)] = 0
    stats["bas"] = st
    teams = {}
    t = np.zeros(n, np.int8)
    team_flag(flag, a0, b0, t)
    teams["flag"] = t
    t = np.zeros(n, np.int8)
    _hold([(r, tm) for r, tm, lab in bas if lab not in DEAD_BAS], a0, b0, t)
    teams["bas_touch"] = t
    t = np.zeros(n, np.int8)
    _hold([(r, tm) for r, tm, ty in nodes if ty and set(ty.split()) - NOT_ON_BALL], a0, b0, t)
    teams["nodes_touch"] = t
    graded = np.zeros(n, bool)
    graded[a0:b0 + 1] = True
    out = {}
    has_flag = np.isin(flag, (0, 1, 2, 3)).any()
    for name, (s, tm) in VARIANTS.items():
        if not has_flag and "flag" in (s, tm):
            continue
        out[name] = (np.where(graded, stats[s], 0).astype(np.int8), np.where(graded, teams[tm], 0).astype(np.int8))
    return out
