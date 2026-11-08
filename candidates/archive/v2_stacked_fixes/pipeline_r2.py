"""v2_stacked_fixes: frozen learned DFL frame evidence + v1's Viterbi + v2's boundary snaps.

ROUND 2 (after the round-2 judges). Shipped pipeline, per graded block:
  evidence  frozen LightGBM models in lde_frozen/ (a snapshot of learned_dfl_evidence's round-1
            features.py, model_ha.txt, model_d.txt, model_c.txt; trained only on the open DFL/IDSSE
            tracking of OTHER games, labelled by its tracking BallPossession / BallStatus flags;
            see lde_frozen/PROVENANCE.md). md5s and the lightgbm version are asserted at load.
    team    team_logit = w_lde * clip(logit p_home, +-lde_clip) + Smart Tagging priors
    dead    dead_logit = logit p_dead (learned) + w_long * long_gated + w_out * out + tag_dead
            long_gated: raw-ball gap > long_s whose mean tracked-player speed < lm_gate;
            out: filled ball beyond a line by > 0.5 m (3-frame majority)
    emission E = temperature * [log(1-pD)+log pH, log(1-pD)+log(1-pH), log pD] (v1's form)
  decode    possession_pipeline.viterbi (v1 costs)
  snaps     D end -> end of the long ball gap within snap_win_s; alive -> D start -> first ball-out
            frame in the previous out_lookback_s
  hygiene   merge_short_v2 (one short-segment merge, shortest first; min_poss_s / min_dead_s 1 s)
  hand-over H <-> A changes moved to the hand-over detector's (model_c) peak within +-peak_snap_s
  tags      (match only, no-op on the bench) pre-kick-off D before each half's 'kick off (start)'

Round-2 build log (dev bench; robustness views from tuning/views.py; DFL-LOGO from
tuning/dfl_logo.py on leave-one-game-out fold models, tracking-flag labels, two degradations r1/r2)
  step                        clean          offset  osd     stress2  DFL-LOGO r1 / r2
  R1 shipped                  0.6468/0.8152  0.6050  0.5855  0.5476   0.4354 / 0.4440 (peak 0.5)
  + hand-over peak snap 0.5 s 0.6513/0.8169  0.6046  0.5864  0.5460   kept (+0.0045, 4/4 rows up)
  + learned dead logit        0.6602/0.8120  0.6088  0.5884  0.5331   0.5447 / 0.5395   kept
  + single merge (no pp.merge_short)  0.6602/0.8120  0.6081  0.5886  0.5329           kept (cleanup)
  + inert code removed        identical bench predictions and match file
Rejected: see README.md. Round 1 / round 0 build logs: pipeline_r1.py, pipeline_r0.py.

numpy + lightgbm (inference with num_threads=1), deterministic, no RNG.
"""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import possession_pipeline as pp  # noqa: E402
from possession_pipeline import fill_short_gaps, moving_mean, runs_of, segments, sigmoid  # noqa: E402

H, A, D = 0, 1, 2

CFG = dict(
    # learned team evidence
    w_lde=6.0, lde_clip=5.0,
    # dead evidence: learned logit + gated long ball gap + ball out
    w_long=4.0, long_s=3.0, lm_gate=2.0, w_out=4.0,
    # D boundary snaps
    snap_gap=1, snap_win_s=4.0, snap_back=1, out_lookback_s=12.0, keep_s=0.5,
    # short-segment merge
    hygiene=1,
    # H <-> A changes snapped to the hand-over detector's peak (0 = off)
    peak_snap_s=0.5,
    # NaN-aware moving mean of the raw ball (half width in frames) for the set-piece restart finder
    ball_smooth_hw=2,
    # Smart Tagging (match only; asserted to be a no-op without tags)
    sp_horizon_s=20.0, sp_missing_s=1.0, sp_move_m=1.0, sp_move_s=0.5, sp_fallback_s=5.0,
    sp_prior_s=10.0, sp_near_m=3.0, w_tag_sp=0.8, goal_win_s=15.0, prekick_s=180.0,
    prekick_tol_s=10.0, w_tag=0.2,
    # overrides of possession_pipeline.PARAMS (decoder costs etc.); none
    pp_over=dict(),
)
if os.environ.get("V2_SWEEP") == "1" and os.environ.get("V2_CFG"):   # sweeps only (opt-in)
    CFG.update(json.loads(os.environ["V2_CFG"]))
    print("[v2] WARNING: V2_CFG override active (V2_SWEEP=1): " + os.environ["V2_CFG"],
          file=sys.stderr)
elif os.environ.get("V2_CFG"):
    print("[v2] WARNING: V2_CFG is set but ignored (set V2_SWEEP=1 to apply it)", file=sys.stderr)

PHASE_POSS = {"build up", "progression"}
PHASE_DEF = {"defending in attacking third", "defending in middle third"}


def _side(team, names):
    """+1 home, -1 away, 0 unknown. Matched only against the meta team names: exact
    (case-insensitive), else a generic fuzzy match (difflib ratio > 0.8, unambiguous)."""
    import difflib
    home_name, away_name = names[0].strip().lower(), names[1].strip().lower()
    team = (team or "").strip().lower()
    if not team or team in ("n/a", "na"):
        return 0.0
    if team == home_name:
        return 1.0
    if team == away_name:
        return -1.0
    rh = difflib.SequenceMatcher(None, team, home_name).ratio()
    ra = difflib.SequenceMatcher(None, team, away_name).ratio()
    if max(rh, ra) > 0.8 and abs(rh - ra) > 0.1:
        return 1.0 if rh > ra else -1.0
    return 0.0


def _restart(raw_seen, bsm, s, horizon, miss_k, move_k, move_m, fallback, n):
    """First frame in [s, s+horizon) where the raw ball reappears after >= miss_k missing frames,
    or a seen stationary ball starts moving (> move_m over move_k frames). Fallback s+fallback."""
    for t in range(max(s, 1), min(n - move_k, s + horizon)):
        if raw_seen[t]:
            if t >= miss_k and not raw_seen[t - miss_k:t].any():
                return t
            if t >= move_k and raw_seen[t - move_k] and raw_seen[t + move_k]:
                before = np.hypot(*(bsm[t] - bsm[t - move_k]))
                after = np.hypot(*(bsm[t + move_k] - bsm[t]))
                if before < 0.5 and after > move_m:
                    return t
    return min(n, s + fallback)


_LDE = {}
LDE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lde_frozen")


def _md5(path):
    with open(path, "rb") as handle:
        return hashlib.md5(handle.read()).hexdigest()


def _load_lde():
    """Load the frozen models once. Fails loudly (rule 3: same input -> same file) if any frozen
    file differs from lde_frozen/MANIFEST.json or lightgbm is not the version they were made with."""
    if "fe" in _LDE:
        return
    import importlib.util
    import lightgbm as lgb
    with open(os.path.join(LDE_DIR, "MANIFEST.json")) as handle:
        manifest = json.load(handle)
    for name, want in manifest["md5"].items():
        got = _md5(os.path.join(LDE_DIR, name))
        assert got == want, "lde_frozen/{} md5 {} != MANIFEST {}".format(name, got, want)
    assert lgb.__version__ == manifest["lightgbm"], \
        "lightgbm {} installed, the frozen models need {}".format(lgb.__version__,
                                                                  manifest["lightgbm"])
    spec = importlib.util.spec_from_file_location("v2_lde_features",
                                                  os.path.join(LDE_DIR, "features.py"))
    fe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe)
    with open(os.path.join(LDE_DIR, "model_meta.json")) as handle:
        meta = json.load(handle)
    _LDE.update(fe=fe, meta=meta,
                d=lgb.Booster(model_file=os.path.join(LDE_DIR, "model_d.txt")),
                ha=lgb.Booster(model_file=os.path.join(LDE_DIR, "model_ha.txt")),
                c=lgb.Booster(model_file=os.path.join(LDE_DIR, "model_c.txt")))


def _lde_logits(players, slot_team, ball, sign_half1, blocks, pitch, fps, clip):
    """Clipped log-odds of p_dead and p_home and the hand-over probability p_change from the
    frozen DFL frame models. Same inference as learned_dfl_evidence.frame_probs: H/A and the
    hand-over detector are averaged with the team-swapped features, so they are exactly
    symmetric."""
    _load_lde()
    fe = _LDE["fe"]
    sign = fe.frame_sign(len(ball), sign_half1, blocks)
    F, fnames, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
    assert fnames == _LDE["meta"]["names"], "feature list differs from the frozen model"
    p_dead = _LDE["d"].predict(F, num_threads=1)
    perm, signs = fe.swap_index(fnames, kinds)
    p_home = 0.5 * (_LDE["ha"].predict(F, num_threads=1)
                    + 1.0 - _LDE["ha"].predict(F[:, perm] * signs, num_threads=1))
    p_change = 0.5 * (_LDE["c"].predict(F, num_threads=1)
                      + _LDE["c"].predict(F[:, perm] * signs, num_threads=1))
    lg = lambda p: np.clip(np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)),
                           -clip, clip)
    return lg(np.asarray(p_dead, float)), lg(np.asarray(p_home, float)), np.asarray(p_change, float)


def build_evidence(players, ball_raw, slot_team, sign_half1, blocks, tags, names, pitch, P, C):
    FPS = pp.FPS
    length, width = pitch
    n = len(ball_raw)
    home, away = slot_team == 0, slot_team == 1

    block_id = np.full(n, -1)
    for i, (_, first, last) in enumerate(blocks):
        block_id[first:last + 1] = i

    gap = int(round(P["ball_gap_fill_s"] * FPS))
    ball = np.stack([fill_short_gaps(ball_raw[:, 0].astype(np.float64), gap),
                     fill_short_gaps(ball_raw[:, 1].astype(np.float64), gap)], 1)
    raw_seen = ~np.isnan(ball_raw[:, 0])
    seen = ~np.isnan(ball[:, 0])

    # smoothed raw ball (set-piece restart finder only)
    hw = int(C["ball_smooth_hw"])
    braw = ball_raw.astype(np.float64)
    if hw > 0:
        bsm = np.stack([moving_mean(braw[:, 0], hw), moving_mean(braw[:, 1], hw)], 1)
        bsm[~raw_seen] = np.nan
    else:
        bsm = braw

    vel = np.full(players.shape, np.nan, np.float32)
    vel[1:] = (players[1:] - players[:-1]) * FPS
    spd = np.sqrt((vel ** 2).sum(-1))
    spd[spd > 12.0] = np.nan
    mean_speed = moving_mean(np.nanmean(spd, axis=1), FPS)

    # Dead-ball cues (hand-built, kept small): ball out of the pitch, long slow raw-ball gaps
    margin = 0.5
    out = seen & ((ball[:, 0] < -margin) | (ball[:, 0] > length + margin) |
                  (ball[:, 1] < -margin) | (ball[:, 1] > width + margin))
    out = moving_mean(out.astype(float), 3) > 0.5

    gaps = []                                  # long raw-ball gaps (start, end)
    long_gated = np.zeros(n, bool)
    for s, e in zip(*runs_of(~raw_seen)):
        if e - s > C["long_s"] * FPS:
            gaps.append((int(s), int(e)))
            if np.nanmean(mean_speed[s:e]) < C["lm_gate"]:
                long_gated[s:e] = True

    # Smart Tagging (match only)
    tag_team = np.zeros(n)
    tag_dead = np.zeros(n)
    kick_restarts = []                          # (tag start, restart) of kick-off set pieces
    if tags:
        dist = np.sqrt(((players - ball[:, None, :].astype(np.float32)) ** 2).sum(-1))
        d_home = np.fmin.reduce(dist[:, home], axis=1)
        d_away = np.fmin.reduce(dist[:, away], axis=1)
        del dist
        phase = np.zeros(n)
        sp_team = np.zeros(n)
        block_first = {i: f for i, (_, f, _) in enumerate(blocks)}
        for code, s, e, team, kind in tags:
            s, e = max(0, s), min(n, max(e, s + 1))
            t = _side(team, names)
            if code in PHASE_POSS:
                phase[s:e] += t
            elif code in PHASE_DEF:
                phase[s:e] -= t
            elif code == "set pieces":
                rs = _restart(raw_seen, bsm, s, int(C["sp_horizon_s"] * FPS),
                              int(C["sp_missing_s"] * FPS), int(C["sp_move_s"] * FPS),
                              C["sp_move_m"], int(C["sp_fallback_s"] * FPS), n)
                bi = block_id[min(max(s, 0), n - 1)]
                if bi < 0:                       # tag starts just before a block: use next block
                    later = [f for f in block_first.values() if f >= s]
                    bstart = min(later) if later else 0
                else:
                    bstart = block_first[bi]
                tag_dead[max(bstart, s - FPS):rs] += 1.0
                if t == 0.0 and rs < n and raw_seen[rs]:
                    if np.nan_to_num(min(d_home[rs], d_away[rs]), nan=99.0) < C["sp_near_m"]:
                        t = 1.0 if d_home[rs] <= d_away[rs] else -1.0
                sp_team[rs:min(e, rs + int(C["sp_prior_s"] * FPS))] += t
                if kind.strip().lower().startswith("kick off (start)"):
                    kick_restarts.append((s, rs))
            elif code == "goals":
                ends = [g for g in gaps if abs(g[1] - e) < C["goal_win_s"] * FPS]
                if ends:
                    g0 = min(ends, key=lambda g: abs(g[1] - e))
                    tag_dead[g0[0]:g0[1]] += 1.0
        # each phase counted once: sign of the net vote, weight w_tag; set-piece prior separate
        tag_team = C["w_tag"] * np.sign(phase) + C["w_tag_sp"] * np.clip(sp_team, -1, 1)
        tag_dead = np.clip(tag_dead, 0, 1)

    lpd, lph, p_change = _lde_logits(players, slot_team, ball_raw, sign_half1, blocks, pitch, FPS,
                                     C["lde_clip"])
    team_logit = tag_team + C["w_lde"] * lph
    dead_logit = lpd + C["w_long"] * long_gated + C["w_out"] * out + P["w_tag_dead"] * tag_dead

    eps = 1e-4
    pd = np.clip(sigmoid(dead_logit), eps, 1 - eps)
    ph = np.clip(sigmoid(team_logit), eps, 1 - eps)
    E = np.stack([np.log(1 - pd) + np.log(ph), np.log(1 - pd) + np.log(1 - ph), np.log(pd)], 1)
    aux = dict(out=out)
    extra = dict(gaps=gaps, kick_restarts=kick_restarts, p_change=p_change)
    return E * P["temperature"], aux, extra


# --------------------------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------------------------

def snap_gap_ends(path, gaps, C, fps):
    """Each D end -> the end of the nearest long raw-ball gap within snap_win_s (if that gap
    ends after the D spell starts)."""
    path = path.copy()
    if not gaps:
        return path
    w = int(C["snap_win_s"] * fps)
    ge = np.array([g[1] for g in gaps])
    for s, e, lab in segments(path):
        if lab != D or e >= len(path):
            continue
        j = int(np.argmin(np.abs(ge - e)))
        if abs(ge[j] - e) <= w and ge[j] > s:
            t = int(ge[j])
            if t > e:
                path[e:t] = D
            elif t < e:
                path[t:e] = path[e]
    return path


def snap_dead_start_back(path, out, C, fps):
    """Each alive->D change: start D at the first ball-out frame in the previous look-back
    (staying keep_s after the possession's start). No fallback shift."""
    path = path.copy()
    segs = segments(path)
    lb, keep = int(C["out_lookback_s"] * fps), int(C["keep_s"] * fps)
    for i in range(1, len(segs)):
        s0, _, l0 = segs[i - 1]
        s, _, l = segs[i]
        if l != D or l0 == D:
            continue
        lo = max(s0 + keep, s - lb)
        if lo >= s:
            continue
        hits = np.where(out[lo:s + fps])[0]
        if len(hits):
            t = lo + int(hits[0])
            if t < s:
                path[t:s] = D
    return path


def merge_short_v2(path, P, fps):
    """Recompute segments after every merge, shortest offending segment first. prev == next ->
    that label; short alive between H and A -> the next team; short alive with exactly one D
    neighbour -> its alive neighbour; D-x-D kept."""
    min_len = {H: int(P["min_poss_s"] * fps), A: int(P["min_poss_s"] * fps),
               D: int(P["min_dead_s"] * fps)}
    path = path.copy()
    while True:
        segs = segments(path)
        if len(segs) == 1:
            break
        cand = sorted((e - s, i) for i, (s, e, lab) in enumerate(segs) if e - s < min_len[lab])
        done = False
        for _, i in cand:
            s, e, lab = segs[i]
            prev = segs[i - 1][2] if i > 0 else None
            nxt = segs[i + 1][2] if i + 1 < len(segs) else None
            if prev is not None and prev == nxt:
                path[s:e] = prev
            elif lab == D or prev is None or nxt is None:
                continue
            elif prev != D and nxt != D:
                path[s:e] = nxt
            else:                                   # exactly one D neighbour
                path[s:e] = nxt if prev == D else prev
            done = True
            break
        if not done:
            break
    return path


def snap_to_peak(path, pc, fps, window_s):
    """Move each direct H <-> A change to the frame of highest hand-over probability within
    +-window_s (staying inside both spells). Same rule as learned_dfl_evidence.snap_to_peak."""
    path = path.copy()
    w = int(window_s * fps)
    segs = segments(path)
    for i in range(1, len(segs)):
        s0, _, l0 = segs[i - 1]
        s, e, l = segs[i]
        if l0 == D or l == D:
            continue
        lo, hi = max(s0 + 1, s - w), min(e - 1, s + w)
        if hi <= lo:
            continue
        t = lo + int(np.argmax(pc[lo:hi]))
        path[min(t, s):max(t, s)] = l if t < s else l0
    return path


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps, cfg=None):
    C = dict(CFG, **(cfg or {}))
    P = dict(pp.PARAMS, **C["pp_over"])
    E, aux, extra = build_evidence(players, ball, slot_team, sign_half1, blocks, tags, names,
                                   pitch, P, C)
    out = np.full(len(ball), 3, np.int8)
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        path = pp.viterbi(E[seg], P)
        if C["snap_gap"]:
            g = [(s - first, e - first) for s, e in extra["gaps"] if s >= first and e <= last + 1]
            path = snap_gap_ends(path, g, C, fps)
        if C["snap_back"]:
            path = snap_dead_start_back(path, aux["out"][seg], C, fps)
        if C["hygiene"]:
            path = merge_short_v2(path, P, fps)
        if C["peak_snap_s"]:
            path = snap_to_peak(path, extra["p_change"][seg], fps, C["peak_snap_s"])
        for s, rs in extra["kick_restarts"]:
            if first - C["prekick_tol_s"] * fps <= s < first + C["prekick_s"] * fps and rs > first:
                path[:min(rs, last + 1) - first] = D
                break
        out[seg] = path
    return out
