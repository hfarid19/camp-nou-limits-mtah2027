"""Formation / role-aware inputs for the possessor-pointer network (r5_ptr_role = r5_ptr_ballc + role features).

prepare(players, slot_team, ball, sign_frames, fps, frames, K=8) -> dict(xs, ms, xf, sel) as ballc.prepare, with
    xs (T, K, FS_R = 10 + 12): ballc's 10 per-player features + 12 role features of the same kept player
    xf (T, FF_R = 14 + 12):    ballc's 14 frame features + 6 shape features of the home team, then 6 of the away team
Everything is computed per frame from the positions of the players tracked in that frame (deterministic, numpy, no
labels, no temporal state), so missing (off-screen) players simply do not count; the visible counts are features.

Own-team orientation: depth d = team * x, lateral l = team * y (canonical / 10 m coordinates, home attacks +x, team
+1 home / -1 away), i.e. every player is described in his own team's attacking direction. A home <-> away swap
(negate coordinates and team flags) therefore leaves every per-player role feature unchanged and exchanges the two
teams' shape blocks (swap below).

Per team and frame, from its visible players:
    goalkeeper-like = the deepest visible player if he is within 22.5 m of his own goal line (d < -3.0) and more than
                      8 m behind the next teammate (or alone)
    outfield        = the other visible players
    lines           = 1-D clustering of the sorted outfield depths: a new line starts where the next depth is more
                      than 5 m further forward (line 0 = the back line)
Role features per player (12):
    0 depth rank among visible teammates / (n - 1)   (0 = deepest, 1 = most advanced)
    1 lateral rank among visible teammates / (n - 1)
    2 line index / (lines - 1)  (0 back line .. 1 front line; 0 for the goalkeeper-like player)
    3 outfield teammates on the same line (including him) / 5   (0 for the goalkeeper-like player)
    4 deepest outfield player (last defender) flag
    5 goalkeeper-like flag
    6 depth - own outfield centroid depth, 7 lateral - centroid lateral, 8 distance to the centroid (clipped +-5)
    9 own team visible / 11, 10 opponents visible / 11
   11 depth - own defensive line (mean depth of line 0), clipped +-10
Shape features per team (6): visible / 11, length (outfield depth range), width (outfield lateral range), defensive
    line height (line 0 mean depth + 5.25 = distance from the own goal line, / 10 m), outfield centroid depth,
    number of lines / 5. All 0 when the team has no visible outfield player.
"""

import numpy as np

import radar_common as rc
import ballc

NR, NG = 12, 6
FS_R, FF_R = ballc.FS_B + NR, ballc.FF_B + 2 * NG
GK_DEPTH, GK_GAP, LINE_GAP = -3.0, 0.8, 0.5           # canonical / 10 m units
GOAL = rc.L_HALF / rc.SCALE                            # own goal line at depth -5.25


def _rank(v, vis):
    """rank of v among the visible entries of each row (stable; invisible rows sort last) -> (T, N) int."""
    order = np.argsort(np.where(vis, v, np.inf), axis=1, kind="stable")
    rk = np.empty_like(order)
    np.put_along_axis(rk, order, np.arange(v.shape[1])[None, :].repeat(len(v), 0), 1)
    return order, rk


def team_roles(dep, lat, vis):
    """one team's slots: dep, lat (T, N) own-oriented depth / lateral, vis (T, N) bool -> R (T, N, NR), G (T, NG)."""
    T, N = dep.shape
    R = np.zeros((T, N, NR))
    G = np.zeros((T, NG))
    if N == 0:
        return R, G
    dep = np.where(vis, dep, 0.0)
    lat = np.where(vis, lat, 0.0)
    n = vis.sum(1)
    den = np.maximum(n - 1, 1)[:, None].astype(np.float64)
    order, rkd = _rank(dep, vis)
    _, rkl = _rank(lat, vis)
    R[..., 0] = np.where(vis, rkd / den, 0.0)
    R[..., 1] = np.where(vis, rkl / den, 0.0)
    rows = np.arange(T)
    sd = np.take_along_axis(np.where(vis, dep, np.inf), order, 1)            # sorted depths, inf padding
    d0 = sd[:, 0]
    d1 = sd[:, 1] if N > 1 else np.full(T, np.inf)
    gk_frame = (n >= 1) & (d0 < GK_DEPTH) & ((n == 1) | (d1 - d0 > GK_GAP))
    gk = np.zeros((T, N), bool)
    gk[rows[gk_frame], order[gk_frame, 0]] = True
    out = vis & ~gk
    no = out.sum(1)
    # lines over the outfield depths
    oorder, _ = _rank(dep, out)
    osd = np.take_along_axis(np.where(out, dep, np.inf), oorder, 1)
    valid = np.arange(N)[None, :] < no[:, None]
    gap = np.zeros((T, N))
    if N > 1:
        with np.errstate(invalid="ignore"):
            gap[:, 1:] = np.where(valid[:, 1:], osd[:, 1:] - osd[:, :-1], 0.0)
    lid_sorted = np.cumsum(gap > LINE_GAP, 1)
    lid = np.zeros((T, N), np.int64)
    np.put_along_axis(lid, oorder, lid_sorted, 1)
    lid = np.where(out, lid, -1)
    nl = np.where(no > 0, lid.max(1) + 1, 0)
    same = (lid[:, :, None] == lid[:, None, :]) & out[:, :, None] & out[:, None, :]
    R[..., 2] = np.where(out, lid / np.maximum(nl - 1, 1)[:, None], 0.0)
    R[..., 3] = np.where(out, same.sum(2) / 5.0, 0.0)
    last = np.zeros((T, N), bool)
    has = no > 0
    last[rows[has], oorder[has, 0]] = True
    R[..., 4] = last
    R[..., 5] = gk
    cnt = np.maximum(no, 1)
    cd = np.where(out, dep, 0.0).sum(1) / cnt
    cl = np.where(out, lat, 0.0).sum(1) / cnt
    ddep, dlat = dep - cd[:, None], lat - cl[:, None]
    R[..., 6] = np.where(vis, np.clip(ddep, -5, 5), 0.0)
    R[..., 7] = np.where(vis, np.clip(dlat, -5, 5), 0.0)
    R[..., 8] = np.where(vis, np.minimum(np.sqrt(ddep ** 2 + dlat ** 2), 5.0), 0.0)
    l0 = out & (lid == 0)
    dline = np.where(l0, dep, 0.0).sum(1) / np.maximum(l0.sum(1), 1)
    R[..., 11] = np.where(vis & has[:, None], np.clip(dep - dline[:, None], -10, 10), 0.0)
    R[~vis] = 0.0
    big = 1e9
    dmax = np.where(out, dep, -big).max(1)
    dmin = np.where(out, dep, big).min(1)
    lmax = np.where(out, lat, -big).max(1)
    lmin = np.where(out, lat, big).min(1)
    G[:, 0] = n / 11.0
    G[:, 1] = np.where(has, dmax - dmin, 0.0)
    G[:, 2] = np.where(has, lmax - lmin, 0.0)
    G[:, 3] = np.where(has, dline + GOAL, 0.0)
    G[:, 4] = np.where(has, cd, 0.0)
    G[:, 5] = nl / 5.0
    return R, G


def roles(players, slot_team, sign_frames, frames):
    """all slots: R (T, S, NR) role features, G (T, 2 NG) home then away shape features."""
    f = np.asarray(frames)
    s = np.asarray(sign_frames, np.float64)[f]
    P = np.asarray(players, np.float64)[f]
    vis = ~np.isnan(P[:, :, 0])
    pc = rc.canon(P, s[:, None])
    team = np.where(np.asarray(slot_team) == 0, 1.0, -1.0)
    dep = np.nan_to_num(pc[..., 0] * team[None, :])
    lat = np.nan_to_num(pc[..., 1] * team[None, :])
    T, S = vis.shape
    R = np.zeros((T, S, NR))
    G = np.zeros((T, 2 * NG))
    nvis = []
    for k, tv in enumerate((1.0, -1.0)):
        ix = np.where(team == tv)[0]
        r, g = team_roles(dep[:, ix], lat[:, ix], vis[:, ix])
        R[:, ix] = r
        G[:, k * NG:(k + 1) * NG] = g
        nvis.append(vis[:, ix].sum(1) / 11.0)
    own = np.where(team[None, :] > 0, nvis[0][:, None], nvis[1][:, None])
    opp = np.where(team[None, :] > 0, nvis[1][:, None], nvis[0][:, None])
    R[..., 9] = np.where(vis, own, 0.0)
    R[..., 10] = np.where(vis, opp, 0.0)
    return R, G


def prepare(players, slot_team, ball, sign_frames, fps, frames, K=8):
    inp = ballc.prepare(players, slot_team, ball, sign_frames, fps, frames, K)
    R, G = roles(players, slot_team, sign_frames, frames)
    sel = inp["sel"]
    Rk = np.take_along_axis(R, np.clip(sel, 0, None)[..., None], 1)
    Rk[sel < 0] = 0.0
    xs = np.concatenate([inp["xs"].astype(np.float64), Rk], -1)
    xf = np.concatenate([inp["xf"].astype(np.float64), G], -1)
    return dict(xs=xs.astype(np.float32), ms=inp["ms"], xf=xf.astype(np.float32), sel=sel)


def swap(xs, xf):
    """team swap: ballc.swap (role features are own-team relative: unchanged) + exchange the two shape blocks."""
    xs, xf = ballc.swap(xs, xf)
    a, b = ballc.FF_B, ballc.FF_B + NG
    xf[..., a:a + NG], xf[..., b:b + NG] = xf[..., b:b + NG].copy(), xf[..., a:a + NG].copy()
    return xs, xf
