"""degrade_v2 plus occlusion-driven player fragmentation, fitted to the competition's tracking.

On BAR-ATH 62% of player tracklets end with another player within 2 m, against 16% under
degrade_v2, whose tracklet cuts are random in time. Tracklet lengths are also heavier-tailed
(p10 / p50 / p90 2.6 / 36 / 285 s vs 9.8 / 59 / 196 s). The automatic tracker loses identities
in crowds, which are where duels and possession changes happen.

degrade_v3 keeps degrade_v2's ball model unchanged and replaces the player model:
- every time a player comes within `occ_r` m of another player, the track breaks there with
  probability `p_occ`; it resumes after an exponential gap of mean `occ_gap_s` in a fresh slot;
- independently, tracks also break at random with mean tracklet length `tracklet_s` (now long),
  followed by a gap of mean `player_gap_s`.
Knobs fitted with fit_degrade_v3.py on label-free statistics only (match_stats).
"""

import numpy as np

import degrade_v2 as dv

FPS = dv.FPS
DEFAULTS = dict(dv.DEFAULTS)
DEFAULTS.update(occ_r=0.5, p_occ=0.82, occ_gap_s=0.2, tracklet_s=900.0, player_gap_s=18.4)


def _nearest_other(players):
    """(N, P) distance to the nearest other player (inf where untracked)."""
    n, P, _ = players.shape
    out = np.full((n, P), np.inf, np.float32)
    for s in range(0, n, 20000):
        x = players[s:s + 20000].astype(np.float32)
        d = np.linalg.norm(x[:, :, None, :] - x[:, None, :, :], axis=3)
        d[np.isnan(d)] = np.inf
        idx = np.arange(P)
        d[:, idx, idx] = np.inf
        out[s:s + 20000] = d.min(2)
    return out


def degrade_players(players, team_of_player, rng, p):
    n = len(players)
    spt = int(p["slots_per_team"])
    slots = np.full((n, 2 * spt, 2), np.nan, np.float32)
    slot_team = np.array([0] * spt + [1] * spt, np.int8)
    sm = np.stack([dv.smooth_runs(players[:, j], p["smooth_frames"]) for j in range(players.shape[1])], 1)
    near = _nearest_other(players) < p["occ_r"]
    for team in (0, 1):
        free_at = np.zeros(spt, np.int64)
        pieces = []
        for j in np.where(team_of_player == team)[0]:
            seen = ~np.isnan(players[:, j, 0])
            close = near[:, j]
            entries = np.where(close[1:] & ~close[:-1])[0] + 1
            breaks = entries[rng.random(len(entries)) < p["p_occ"]]
            for s, e in zip(*dv.pp.runs_of(seen)):
                t = s
                while t < e:
                    stop = min(e, t + dv._exp_frames(rng, p["tracklet_s"]) + FPS)
                    gap = dv._exp_frames(rng, p["player_gap_s"])
                    k = np.searchsorted(breaks, t + FPS // 5)            # next occlusion break
                    if k < len(breaks) and breaks[k] < stop:
                        stop = int(breaks[k])
                        gap = dv._exp_frames(rng, p["occ_gap_s"])
                    pieces.append((t, stop, j))
                    t = stop + max(1, gap)
        pieces.sort()
        for s, e, j in pieces:
            free = np.where(free_at <= s)[0]
            if not len(free):
                continue
            kk = int(rng.choice(free))
            slots[s:e, team * spt + kk] = sm[s:e, j]
            free_at[kk] = e + int(rng.integers(1, 50))
    return slots, slot_team


def degrade_v3(players, team_of_player, ball, status, graded, rng, params=None):
    p = dict(DEFAULTS)
    p.update(params or {})
    slots, slot_team = degrade_players(players, team_of_player, rng, p)
    out = dv.smooth_runs(ball, p["ball_smooth_frames"]).astype(np.float32)
    out[dv.hide_ball(ball, players, status, graded, rng, p)] = np.nan
    return slots, slot_team, out
