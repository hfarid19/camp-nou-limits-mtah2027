"""Ball-centred inputs for the possessor-pointer network (round 5, R5-3 variant r5_ptr_ballc).

prepare(players, slot_team, ball, sign_frames, fps, frames, K=8) -> dict
    xs  (T, K, FS_B) the K tracked players nearest to the ball, nearest first, coordinates relative to the ball in
        the attack frame (home attacks +x): [dx, dy (player - ball) / 10 m clipped +-10, vx, vy / 5 m/s, team +-1,
        distance / 10 m, exp(-distance / 2 m), ball-seen flag, speed / 5 m/s, rank / K]
    ms  (T, K) 1 where the slot is filled (fewer than K tracked players -> padding)
    xf  (T, FF_B) radar_common.prepare's 10 frame features (interpolated ball position in the attack frame, seen flag,
        log time since / until the ball was seen, players tracked, ball speed, players' mean speed, gap length, bias)
        + ball kinematics: vx, vy (attack frame, / 20 m/s), acceleration (m/s^2 / 20), direction change |angle| / pi
    sel (T, K) int the slot index of each kept player (-1 for padding)
Ball where it is missing = radar_common.interp_ball (linear interpolation between the last and the next observed
ball), exactly what the pipeline has at inference; no network-inferred (radar) ball is used.
swap(xs, xf): team swap = negate dx, dy, vx, vy, team and the frame ball position / velocity.
"""

import numpy as np

import radar_common as rc

FS_B, FF_B = 10, 14


def _kinematics(ball_i, sg, fps):
    """ball velocity (attack frame), acceleration, turn angle from the interpolated ball at full rate."""
    n = len(ball_i)
    v = np.zeros((n, 2))
    v[1:-1] = (ball_i[2:] - ball_i[:-2]) * fps / 2.0
    v = rc._mmean_axis0(v, 2)
    sp = np.sqrt((v ** 2).sum(1))
    acc = np.zeros(n)
    h = max(1, int(round(0.2 * fps)))
    acc[h:-h] = (sp[2 * h:] - sp[:-2 * h]) / (2 * h / fps)
    turn = np.zeros(n)
    a, b = v[:-2 * h], v[2 * h:]
    na, nb = np.sqrt((a ** 2).sum(1)), np.sqrt((b ** 2).sum(1))
    ok = (na > 0.5) & (nb > 0.5)
    cos = np.where(ok, (a * b).sum(1) / np.maximum(na * nb, 1e-9), 1.0)
    turn[h:-h] = np.arccos(np.clip(cos, -1, 1)) / np.pi
    return v * sg[:, None], acc, turn


def prepare(players, slot_team, ball, sign_frames, fps, frames, K=8):
    f = np.asarray(frames)
    inp = rc.prepare(players, slot_team, ball, sign_frames, fps, f)
    sg = np.asarray(sign_frames, np.float64)
    bi = rc.interp_ball(ball)
    vb, acc, turn = _kinematics(bi, sg, fps)
    pos = inp["pos"].astype(np.float64)                       # (T, S, 2) canonical / 10 m (0 where empty)
    ms = inp["ms"] > 0
    bc = inp["xf"][:, 0:2].astype(np.float64)                 # interpolated ball, canonical / 10 m
    rel = pos - bc[:, None, :]
    d = np.sqrt((rel ** 2).sum(-1))
    d = np.where(ms, d, np.inf)
    T, S = d.shape
    Kk = min(K, S)
    order = np.argsort(d, axis=1, kind="stable")[:, :Kk]
    dk = np.take_along_axis(d, order, 1)
    mk = np.isfinite(dk)
    xs_all = inp["xs"].astype(np.float64)
    g = lambda a: np.take_along_axis(a, order[..., None], 1)  # noqa: E731
    relk, velk, teamk = g(rel), g(xs_all[..., 2:4]), g(xs_all[..., 4:5])[..., 0]
    seenk, spdk = g(xs_all[..., 9:10])[..., 0], g(xs_all[..., 10:11])[..., 0]
    xs = np.zeros((T, K, FS_B))
    dkf = np.where(mk, dk, 0.0)
    xs[:, :Kk, 0:2] = np.clip(relk, -10, 10)
    xs[:, :Kk, 2:4] = velk
    xs[:, :Kk, 4] = teamk
    xs[:, :Kk, 5] = np.minimum(dkf, 10.0)
    xs[:, :Kk, 6] = np.exp(-dkf * 10.0 / 2.0)
    xs[:, :Kk, 7] = seenk
    xs[:, :Kk, 8] = spdk
    xs[:, :Kk, 9] = np.arange(Kk)[None, :] / K
    msk = np.zeros((T, K))
    msk[:, :Kk] = mk
    xs[msk == 0] = 0.0
    sel = -np.ones((T, K), np.int64)
    sel[:, :Kk] = np.where(mk, order, -1)
    xf = np.zeros((T, FF_B))
    xf[:, :10] = inp["xf"]
    xf[:, 10:12] = np.clip(vb[f] / 20.0, -3, 3)
    xf[:, 12] = np.clip(acc[f] / 20.0, -3, 3)
    xf[:, 13] = turn[f]
    return dict(xs=xs.astype(np.float32), ms=msk.astype(np.float32), xf=xf.astype(np.float32), sel=sel)


def swap(xs, xf):
    xs = xs.copy()
    xs[..., 0:5] *= -1.0
    xf = xf.copy()
    xf[..., [0, 1, 10, 11]] *= -1.0
    return xs, xf
