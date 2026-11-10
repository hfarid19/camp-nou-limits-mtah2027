"""degrade_v4: degrade_v2's player model with a dead-ball model refitted to the competition match.

Why (candidates/_scratch/v5b_calib*.py, all label-free on the match): the four starter baselines
(starter/baseline.py) score far better on the degrade_v2 bench than the organisers' published
full-match numbers (EVALUATION.md): raw 0.728 / 0.103 vs 0.653 / 0.049 (macroF1 / spellF1),
smooth25 0.740 / 0.289 vs 0.675 / 0.200, smooth50 0.729 / 0.301 vs 0.672 / 0.243, smooth125
0.666 / 0.212 vs 0.636 / 0.224. The gap shrinks with the smoothing window, so it is short-scale:
on the match the ball is visible in short islands during stoppages. Compared with degrade_v2:

- the match ball has 49 visible runs of 1-10 s per 90 min (dfl2: 23), 52% of them with the
  players at dead-ball speed (dfl2: 22%). In those slow islands the ball is INSIDE the pitch
  (93%), 1.7 m from the nearest player and moving at about 2.8 m/s: a ball carried, bounced or
  rolled by players during the stoppage, not the frozen out-of-bounds ball the clean DFL feed
  shows (99% static). degrade_v2 revealed that frozen ball in 22% of dead runs and in 0.3 s
  splits, which is a trivially recognisable "dead" cue the match never shows;
- the match ball has almost no gaps of 0.5-5 s (1.7 per 90; dfl2 22) and many of 5-10 s
  (45 per 90; dfl2 18): a lost ball stays lost for at least ~5 s, and stoppages are cut into
  several gaps by the islands;
- when the ball goes out it is tracked for ~0.8 s past the line while moving (69 out-runs per
  90, median 0.84 s, 10% static); degrade_v2 hid it 2.2 s before the dead flag instead.

v4 therefore, per dead run: the ball keeps rolling (extrapolated) for `roll_s` after the flag,
is hidden, then re-appears in islands (Poisson rate `isl_per_min` per minute of hidden dead
time, lognormal length `isl_s` / `isl_sigma`) carried by a player near the restart spot (offset
`isl_offset_m`, random walk `isl_step_m`), is carried to the restart spot for `reveal_s` and parked
there (frozen, as the clean feed shows it) for `park_s` before the restart. Open-play gaps are long (`alive_gap_min_s` + exponential `alive_gap_s`), plus trivial
micro-gaps. Knobs fitted by fit_degrade_v4.py to the published baseline scores (label-free: the
match's own tracking plus organiser-published numbers; no ground truth read) and to the match's
run statistics. Players: degrade_v2.degrade_players unchanged.
"""

import numpy as np

import degrade_v2 as dv
import possession_pipeline as pp

FPS = dv.FPS
DEFAULTS = dict(dv.DEFAULTS)
DEFAULTS.update(
    # Set 2026-11-09 from the match's label-free run statistics (fit_degrade_v4.py --check; the
    # random search in fit_degrade_v4.py did not find this basin, see README): gap-length and
    # visible-run histograms, slow-island share, island ball speed and distance to the carrier,
    # static-ball share, ball missing 0.41.
    roll_s=0.6, p_roll=0.4, roll_decay=0.96, reveal_s=2.5, park_s=0.6,
    isl_per_min=2.0, isl_s=3.5, isl_sigma=0.8, isl_first_s=5.0, isl_offset_m=1.5, isl_step_m=0.15,
    isl_carrier_r=25.0,
    alive_gaps_per90=15.0, alive_gap_min_s=4.5, alive_gap_s=2.5, crowd_boost=3.0,
    micro_per90=13.0, micro_frames=3, fill_gap_s=5.0, pitch=(105.0, 68.0), park_inset_m=0.2,
)
# knobs of the v2 model that v4 does not use
for _k in ("lead_s", "p_keep_dead", "p_split", "split_s"):
    DEFAULTS.pop(_k, None)


def _exp(rng, mean_s):
    return int(rng.exponential(mean_s * FPS)) if mean_s > 0 else 0


def _lognormal_frames(rng, mean_s, sigma):
    mu = np.log(max(mean_s, 1e-3)) - sigma ** 2 / 2
    return max(FPS // 2, int(rng.lognormal(mu, sigma) * FPS))


def _carried(rng, players, spot, a, b, p):
    """Ball positions over [a, b) carried by a player chosen near `spot` (None if nobody is near)."""
    d = np.linalg.norm(players[a] - spot, axis=1)
    cand = np.where(np.nan_to_num(d, nan=np.inf) < p["isl_carrier_r"])[0]
    if not len(cand):
        return None
    w = 1.0 / (np.nan_to_num(d[cand], nan=np.inf) + 3.0)
    j = int(rng.choice(cand, p=w / w.sum()))
    off = rng.normal(0, p["isl_offset_m"], 2)
    steps = rng.normal(0, p["isl_step_m"], (b - a, 2)).cumsum(0)
    traj = players[a:b, j] + off + steps
    bad = ~np.isfinite(traj[:, 0])                  # hold the last position where the carrier is untracked
    if bad.any():
        idx = np.where(~bad, np.arange(b - a), 0)
        np.maximum.accumulate(idx, out=idx)
        traj = traj[idx]
        traj[~np.isfinite(traj[:, 0])] = spot
    return traj


def dead_ball(ball, players, status, graded, rng, p):
    """(ball_out (N, 2) float32 with NaN where hidden). Ball model only (players: clean DFL)."""
    n = len(ball)
    out = ball.astype(np.float32).copy()
    hidden = np.zeros(n, bool)
    dead = (status == 0) & graded
    alive = (status == 1) & graded
    for s, e in zip(*pp.runs_of(dead)):
        # before the restart: the ball is carried to the spot for `reveal_s`, then parked `park_s`
        park = min(e - s, _exp(rng, p["park_s"]))
        reveal = min(e - s - park, _exp(rng, p["reveal_s"]))
        h0, h1 = s, e - reveal - park                # hidden stretch (before islands)
        if park > 0:                                 # the parked ball is tracked at the line, not beyond it
            L, Wd = p["pitch"]
            m = p["park_inset_m"]
            out[e - park:e, 0] = np.clip(out[e - park:e, 0], m, L - m)
            out[e - park:e, 1] = np.clip(out[e - park:e, 1], m, Wd - m)
        if reveal > 0:
            spot = ball[e - 1] if np.all(np.isfinite(ball[e - 1])) else ball[s - 1]
            traj = _carried(rng, players, spot, h1, h1 + reveal, p)
            if traj is not None:
                out[h1:h1 + reveal] = traj
        if rng.random() < p["p_roll"] and s >= 2:
            k = min(_exp(rng, p["roll_s"]), max(h1 - s, 0))
            v = ball[s - 1] - ball[s - 2]
            if np.all(np.isfinite(v)):
                pos = ball[s - 1].astype(np.float64).copy()
                for t in range(k):
                    v = v * p["roll_decay"]
                    pos = pos + v
                    out[s + t] = pos
            h0 = s + k
        hidden[h0:h1] = True
        # islands: the ball carried by a player near the restart spot
        t = h0 + int(p["isl_first_s"] * FPS)
        rate = p["isl_per_min"] / 60.0
        while t < h1:
            a = t + (_exp(rng, 1.0 / rate) if rate > 0 else n)
            if a >= h1:
                break
            L = _lognormal_frames(rng, p["isl_s"], p["isl_sigma"])
            b = min(a + L, h1)
            spot = ball[a] if np.all(np.isfinite(ball[a])) else ball[s - 1]
            traj = _carried(rng, players, spot, a, b, p)
            if traj is not None:
                out[a:b] = traj
                hidden[a:b] = False
            t = b
    # long gaps in open play (ball lost in a crowd / in the air), more likely in a crowd
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    crowd = (np.nan_to_num(d, nan=99.0) < 3.0).sum(1) >= 3
    rate = p["alive_gaps_per90"] / (90 * 60 * FPS) / max(alive.mean() / max(graded.mean(), 1e-9), 1e-9)
    w = np.where(crowd, p["crowd_boost"], 1.0)
    w *= 1.0 / max(np.mean(w[alive]), 1e-9)
    for s in np.where(alive & (rng.random(n) < rate * w))[0]:
        hidden[s:s + int(p["alive_gap_min_s"] * FPS) + _exp(rng, p["alive_gap_s"])] = True
    # the match tracker shows almost no gaps of 0.5-5 s: a short loss is bridged by interpolation
    fill = int(p["fill_gap_s"] * FPS)
    for a, b in zip(*pp.runs_of(hidden)):
        if b - a < fill and a > 0 and b < n and np.all(np.isfinite(out[a - 1])) and np.all(np.isfinite(out[b])):
            w = np.linspace(0, 1, b - a + 2)[1:-1, None]
            out[a:b] = out[a - 1] * (1 - w) + out[b] * w
            hidden[a:b] = False
    mrate = p["micro_per90"] / (90 * 60 * FPS) / max(graded.mean(), 1e-9)
    for s in np.where(graded & (rng.random(n) < mrate))[0]:
        hidden[s:s + int(rng.integers(1, int(p["micro_frames"]) + 1))] = True
    out = dv.smooth_runs(out, p["ball_smooth_frames"]).astype(np.float32)
    out[hidden & graded] = np.nan
    return out


def degrade_v4(players, team_of_player, ball, status, graded, rng, params=None):
    p = dict(DEFAULTS)
    p.update(params or {})
    slots, slot_team = dv.degrade_players(players, team_of_player, rng, p)
    dball = dead_ball(ball, players, status, graded, rng, p)
    return slots, slot_team, dball
