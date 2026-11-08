"""degrade_v2: make clean tracking look like the competition's automatic tracking, fitted to it.

tune_on_sample_data.degrade (v1) hides the ball in many short random gaps during play and adds
0.2 m of noise. The competition tracking looks nothing like that (candidates/match_stats.py on
BAR-ATH): the ball is smooth (0.009 m jitter), and 99.7% of its missing frames sit in ~127 long
gaps per 90 min (median 10.7 s), during which the players move like DFL players at dead ball
(median team speed 1.0 m/s vs 1.02 at DFL dead ball, 2.27 alive). Player tracks are smooth and
long (median tracklet 35 s, ~10.25 of 11 players visible per team).

v2 therefore needs the ball status of the clean game, which the DFL data has (BallStatus flag):

- ball: for each dead-ball run, hide the ball from a little before it starts (`lead_s`) to a
  little before the restart (`reveal_s`, the ball becomes visible again as it is placed). A dead
  run is left visible with probability `p_keep_dead`, and a hidden one is split by a short
  visible window with probability `p_split`. Short gaps in open play are added at
  `alive_gaps_per90`, more often when the ball is in a crowd (as in v1). No noise is added (the
  clean DFL ball is already about as smooth as the competition's); a `ball_smooth_frames`
  moving average matches the competition ball's acceleration.
- players: each player's continuous track is cut into tracklets of mean `tracklet_s` with a gap
  of mean `player_gap_s` between them (slots reused like v1), then every position is low-passed
  with a centred moving average of `smooth_frames` frames.

Every knob was fitted by matching label-free statistics of BAR-ATH (fit_degrade_v2.py); no
label of the competition game is used anywhere.
"""

import numpy as np

import possession_pipeline as pp

FPS = 25
# Fitted 2026-11-07 by fit_degrade_v2.py (300 + 100 random-search evaluations on 3 DFL games),
# ball_smooth_frames picked separately (3 matches BAR-ATH's median ball acceleration, 5.6 vs 5.7).
DEFAULTS = dict(lead_s=2.2, reveal_s=0.0, p_keep_dead=0.22, p_split=0.34, split_s=0.3,
                alive_gaps_per90=24.5, alive_gap_s=0.55, crowd_boost=3.0,
                tracklet_s=88.0, player_gap_s=7.3, slots_per_team=21, smooth_frames=13,
                ball_smooth_frames=3)


def _exp_frames(rng, mean_s):
    return int(rng.exponential(mean_s * FPS)) if mean_s > 0 else 0


def hide_ball(ball, players, status, graded, rng, p):
    """Boolean mask of frames where the ball is hidden."""
    n = len(ball)
    hidden = np.zeros(n, bool)
    dead = (status == 0) & graded
    for s, e in zip(*pp.runs_of(dead)):
        if rng.random() < p["p_keep_dead"]:
            continue
        a = max(0, s - _exp_frames(rng, p["lead_s"]))
        b = max(a + 1, e - _exp_frames(rng, p["reveal_s"]))
        hidden[a:b] = True
        if rng.random() < p["p_split"] and b - a > 4 * FPS:
            w = max(1, _exp_frames(rng, p["split_s"]))
            c = int(rng.integers(a + FPS, b - FPS - w)) if b - a - 2 * FPS - w > 0 else a + FPS
            hidden[c:c + w] = False
    # short gaps in open play, more likely when the ball is in a crowd
    alive = (status == 1) & graded
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    crowd = (np.nan_to_num(d, nan=99.0) < 3.0).sum(1) >= 3
    rate = p["alive_gaps_per90"] / (90 * 60 * FPS) / max(alive.mean() / max(graded.mean(), 1e-9), 1e-9)
    w = np.where(crowd, p["crowd_boost"], 1.0)
    w *= 1.0 / max(np.mean(w[alive]), 1e-9)
    starts = np.where(alive & (rng.random(n) < rate * w))[0]
    for s in starts:
        hidden[s:s + max(1, _exp_frames(rng, p["alive_gap_s"]))] = True
    return hidden & graded


def smooth_runs(xy, k):
    """Centred moving average of odd length k over each run of observed frames (edges padded)."""
    k = 2 * (int(k) // 2) + 1
    out = xy.astype(np.float64)
    if k <= 1:
        return out
    kern = np.ones(k) / k
    for s, e in zip(*pp.runs_of(~np.isnan(xy[:, 0]))):
        if e - s > k:
            for c in range(2):
                x = xy[s:e, c].astype(np.float64)
                pad = np.concatenate([np.full(k // 2, x[0]), x, np.full(k // 2, x[-1])])
                out[s:e, c] = np.convolve(pad, kern, mode="valid")
    return out


def degrade_players(players, team_of_player, rng, p):
    """Anonymous, fragmented, smoothed slots (as v1, with v2's tracklet / gap / smoothing knobs)."""
    n = len(players)
    spt = int(p["slots_per_team"])
    slots = np.full((n, 2 * spt, 2), np.nan, np.float32)
    slot_team = np.array([0] * spt + [1] * spt, np.int8)
    sm = np.stack([smooth_runs(players[:, j], p["smooth_frames"]) for j in range(players.shape[1])], 1)
    for team in (0, 1):
        free_at = np.zeros(spt, np.int64)
        pieces = []
        for j in np.where(team_of_player == team)[0]:
            seen = ~np.isnan(players[:, j, 0])
            for s, e in zip(*pp.runs_of(seen)):
                t = s
                while t < e:
                    length = _exp_frames(rng, p["tracklet_s"]) + FPS
                    pieces.append((t, min(e, t + length), j))
                    t += length + _exp_frames(rng, p["player_gap_s"])
        pieces.sort()
        for s, e, j in pieces:
            free = np.where(free_at <= s)[0]
            if not len(free):
                continue
            kk = int(rng.choice(free))
            slots[s:e, team * spt + kk] = sm[s:e, j]
            free_at[kk] = e + int(rng.integers(1, 50))
    return slots, slot_team


def degrade_v2(players, team_of_player, ball, status, graded, rng, params=None):
    p = dict(DEFAULTS)
    p.update(params or {})
    slots, slot_team = degrade_players(players, team_of_player, rng, p)
    out = smooth_runs(ball, p["ball_smooth_frames"]).astype(np.float32)
    out[hide_ball(ball, players, status, graded, rng, p)] = np.nan
    return slots, slot_team, out
