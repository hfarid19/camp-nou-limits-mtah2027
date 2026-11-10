"""Input perturbations for test-time averaging (idea 2): smooth AR(1) position offsets, fixed seeds.

jitter(players, ball, r, fps): pass r uses numpy default_rng(1000 + r); every player slot gets an AR(1)
offset per axis (time constant TAU_S, stationary sd SD_P metres) and the ball one with sd SD_B. NaNs stay
NaN. Deterministic numpy (a sequential recursion; no threaded reductions).
"""

import numpy as np

TAU_S, SD_P, SD_B = 1.0, 0.5, 0.3


def ar1(n, shape, sd, rng, fps):
    a = np.exp(-1.0 / (TAU_S * fps))
    eps = rng.normal(0.0, sd * np.sqrt(1 - a * a), (n,) + shape)
    out = np.empty_like(eps)
    out[0] = rng.normal(0.0, sd, shape)
    for t in range(1, n):
        out[t] = a * out[t - 1] + eps[t]
    return out


def jitter(players, ball, r, fps):
    rng = np.random.default_rng(1000 + r)
    n = len(ball)
    p = players.astype(np.float64) + ar1(n, players.shape[1:], SD_P, rng, fps)
    b = ball.astype(np.float64) + ar1(n, ball.shape[1:], SD_B, rng, fps)
    return p.astype(players.dtype), b.astype(ball.dtype)
