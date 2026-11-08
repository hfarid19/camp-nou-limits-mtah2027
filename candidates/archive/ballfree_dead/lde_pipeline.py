"""learned_dfl_evidence: LightGBM H/A/D frame evidence learned on open DFL tracking + Viterbi.

README
------
Evidence model. Two LightGBM boosters (model_d.txt: dead vs alive; model_ha.txt: home vs away
while alive) on ~190 identity-free, team-relative per-frame features (features.py: ball gap
context, nearest-player distances / counts per team, team shape, last confident control,
restart patterns, multi-scale temporal context). They are trained ONLY on externally labelled
tracking of OTHER games: the open DFL / IDSSE dataset (7 Bundesliga matches, per-frame
BallPossession and BallStatus flags from the tracking), degraded with domain randomisation to
look like the competition's automatic tracking (make_dataset.py; since round 1 half of the
degradations also low-pass the ball, because the competition ball is smooth). Nothing is trained
on Metrica or on the competition match; the bench only picks a few decoder scalars below.

Data source and licence. Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). "An integrated
dataset of spatiotemporal and event data in elite soccer." Scientific Data 12. figshare article
28196177, https://doi.org/10.6084/m9.figshare.28196177 . Data (c) DFL Deutsche Fussball Liga /
IDSSE, licensed CC BY 4.0. Downloaded by fetch_idsse.py (raw data is cached outside the repo and
never committed; see .gitignore).

Rebuild:  python -I fetch_idsse.py --games 7 ; python -I make_dataset.py ; python -I train.py
(data cache: $IDSSE_CACHE or ./external/idsse; figshare version 1 pinned, md5 per file checked
against idsse_manifest.json). Dependencies pinned in requirements.txt: lightgbm 4.7.0 (asserted at
load against model_meta.json), numpy 2.5.4, pandas 3.1.1; deterministic with num_threads=1.

A third booster (model_c.txt) is a hand-over detector: P(a direct H<->A change of the
tracking possession flag within +-4 frames), trained on alive frames with swap augmentation.
Rounds of all three picked by leave-one-DFL-game-out log-loss (round 1 models: d 497, ha 434,
c 321 rounds; LOGO log-loss 0.0567 / 0.2754 / 0.1233). Inference averages each prediction with the prediction on
the team-swapped features, so H/A are exactly symmetric.

DFL label diagnostics (7 games, BallStatus/BallPossession): 412 possession spells per 90 min,
dead share 0.448; spell duration p10/p25/p50/p75/p90 = 0.8/1.7/3.7/8.5/18.6 s (12.5% < 1 s);
stoppage duration p10/p50/p90 = 5.8/19.6/49.4 s. BallPossession keeps the last team while the
ball is in flight and flips on touches, so it has more short spells than event-derived labels.

Decoder (round 3 below supersedes the post-processing and team-logit details here).
E[:,H] = tau*(log pH + log(1-pD)), E[:,A] = tau*(log(1-pH) + log(1-pD)),
E[:,D] = tau*log pD, where the dead log-odds get +w_gap (4) on raw-ball gaps > 3 s whose mean
tracked-player speed is < 2 m/s (the learned dead model under-used long untracked stoppages);
possession_pipeline.viterbi with switch / dead costs, merge_short,
H<->A boundaries snapped to the hand-over detector's peak within 1.0 s (snap_mode 1; the
first-hit rule of possession_pipeline.snap_boundaries is snap_mode 0), then gap-edge D snaps
(a D end moved to the end of the ball gap it sits in, within 4 s). Smart Tagging priors (match
only, a no-op on the bench) are kept small: +1 dead log-odds on the long ball gap whose end is
nearest a goal tag's end (within 15 s; a goal tag runs to the restart kick-off), build up /
progression team prior +-0.4 log-odds (team strings matched exactly against the meta names), and
D before each half's 'kick off (start)' tag (up to the restart if it falls inside the tag, else
up to the tag start). The decoder constants below can be overridden by LDE_<KEY> environment
variables only when LDE_SWEEP=1 (tuning); the default run ignores the environment.

Selection (dev bench only): tau is a no-op because the costs scale with it (Viterbi is scale
invariant), so the choice was switch_cost x dead_cost x beta (x gamma). Plateau: switch
0.5-1.0, dead 4-6 all within 0.611-0.613; picked the centre 0.75 / 5. beta 0.3 (blend v1's
in-match shape logit) cost 0.008 spellF1, so beta 0. gamma (hand-over detector on the switch
cost) was +-0.001 on the bench and on DFL, so 0 (the detector is still used for snapping).

Round 1 (judges' feedback; bench after each step, kept only if > +0.003 or a rules fix):
  w_gap 4 gated long-gap dead term             0.6129 -> 0.6332  kept
  refine_window_s 1.5 -> 1.0                    0.6332 -> 0.6400  kept
  pre-kick-off D, goal prior on the goal gap, exact team names, LDE_SWEEP gate: bench no-op, kept
    (match: ~8 spurious pre-kick-off spells removed, no D forced on live play after kick-offs)
  low-passed-ball augmentation + retrain        0.6400 -> 0.6467  kept (also +0.005 under the
    match-like stress 'offset+smooth+drop' and +0.005 under the AR(1) ball offset)
  rejected: HTD hand-built dead logit blend (<= 0.636), switch x dead re-grid (flat, +-0.002),
    gamma 1-2 (+0.006 clean but -0.003 stress), refine 0.75 (+0.004 clean only), dead-prior
    removal / bias (flat), high-pass team logit to split sandwiches (+0.017 clean, -0.03 stress).
  DFL leave-one-game-out with the round-1 fold models (seed 21, diagnostic only): spellF1 0.541,
  macroF1 0.894 (round 0: 0.527 / 0.893).
Round 2 (judges' feedback; views clean / osd / offset / s2 = judges' OOD stress, + DFL LOGO):
  refine_window_s 1.0 -> 0.5                    0.6467 -> 0.6508  kept (DFL LOGO +0.005 too)
  soft long-gap ramp (gap_soft 0.2, centre 1.75 m/s, length ramp 2-5 s) instead of the hard
    gate: clean flat, s2 +0.006, match dead share 0.477 -> 0.471    kept (robustness)
  model_d / model_c retrained with extra short ball dropouts + stoppage blackouts (match-like
    long gaps, player collapse); model_ha unchanged: clean 0.6516, osd +0.005, offset +0.007,
    s2 +0.007, DFL LOGO +0.004 on blackout degradation           kept (robustness)
  w_gap 4 -> 3 (dead model now learns gaps; flat on all views)    kept (calibration)
  final: dev 0.6502 / 0.8118, osd 0.5837, offset 0.6095, s2 0.5445; match 308 poss, 47.4% D.
  rejected: wide-offset (sd 2 m, tau 1-3 s) retrain of all models (clean -0.02, the team model),
    w_gap 2 / 0, gamma 1, switch 0.5 / 1.0, d_end_snap 0 / 2, min_poss 0.5 / 0.75 (all noise
    or worse). Model files are sha256-checked against model_meta.json at load.
Round 3 (judges' feedback; views clean / offset / osd / s2 + DFL LOGO = bench --dfl, seed 2027,
fold models; kept if dev > +0.003 with macroF1 within 0.01, or a judge-raised rules/robustness fix):
  step                                   clean/macro    offset  osd     s2      DFL
  round 2                                0.6502/0.8118  0.6095  0.5888  0.5387  0.5929
  merge_short_v2 (shortest first, recomputed after each merge; replaces pp.merge_short's
    order-dependent 5-pass walk)          0.6521/0.8116  0.6130  0.5916  0.5414  0.5933  kept
  v2's D post chain: D end -> nearest long ball-gap end within snap_win_s (both ways) and
    alive->D start back to the first out frame (out_lookback_s, keep_s), replacing snap_dead_edges;
    snap_win_s 4 -> 1 and look-back 6 picked where dev is flat and DFL rises (DFL D-end |offset|
    > 1 s 0.198 -> 0.095, p90 +1.60 -> +0.68 s)
                                         0.6547/0.8135  0.6160  0.5954  0.5469  0.6036  kept
  local-contrast team logit l + 1.25 * deadzone(l - moving_mean(l, 100 frames), 1.25)
    (short opposing dips now cross 0; centre of the beta = gate 1.0-1.5 plateau)
                                         0.6725/0.8145  0.6204  0.5882  0.5436  0.6243  kept
  w_tag_team 0.4 -> 0.1 (untestable on the bench; it only merged short opposing touches inside
    ~18 s phase tags): bench no-op, match 315 -> 325 possessions          kept (robustness)
  fail-closed model loading (model_c mandatory, sha256 must cover d/ha/c, lgb_version required),
    stderr warning on any LDE_* variable, make_submission refuses LDE_* : no-op  kept (rules)
  rejected: w_out 4 (+0.0006), contrast hs pre-smoothing (worse), refine_window_s 0.25 / 0.35 /
    0.75 (+-0.002: plateau confirmed), switch 0.5 / 1.0, min_poss 0.6, gamma 1 (all noise or
    worse), w_gap 1.5 / 0 (dev -0.001 / -0.004; DFL +0.001 / +0.0035), split tau (OOF DFL Platt
    slopes 0.95 dead / 0.97 team: nothing to split), temporal short-touch stacker on OOF DFL
    probabilities (clean +0.0016, DFL -0.014 when trained on the candidate's own degradation).
Results (round 0, kept for reference). Dev bench spellF1 0.6129, macroF1 0.7988 (v1 0.4749 /
0.7453), every row +0.12..0.15.
Ablation on the same decoder (switch 0.75): learned dead + v1 team 0.5226; v1 dead + learned
team 0.5613; v1 both 0.4856; learned both 0.6129. DFL leave-one-game-out (degraded with an
unseen seed 21, fold models, tracking-flag labels): spellF1 0.527, macroF1 0.893 (robustness
diagnostic only). Real match: 317 possessions, 43.7% dead. Two --match runs are byte-identical.
"""

import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import possession_pipeline as pp  # noqa: E402
import lde_features as fe         # noqa: E402

warnings.filterwarnings("ignore")
H, A, D = 0, 1, 2

CFG = dict(
    tau=0.15,             # evidence temperature
    switch_cost=0.75,     # H <-> A (at tau 0.15; scaled with tau). Bench plateau 0.5-1.0
    dead_cost=5.0,        # alive <-> D (at tau 0.15; scaled with tau)
    beta=0.0,             # blend weight of v1's in-match team-shape logit
    gamma=0.0,            # weight of the learned hand-over detector on the switch cost
    snap_mode=1.0,        # 0: first-hit snap of H<->A changes, 1: snap to the hand-over peak
    min_poss_s=1.0,
    min_dead_s=1.0,
    refine_window_s=0.5,  # hand-over peak snap window (1.5 -> 1.0 r1: +0.007; -> 0.5 r2: +0.004,
                          # and +0.005 on DFL leave-one-game-out)
    control_radius_m=1.5,
    d_end_snap_s=4.0,     # move a D end to the end of the ball gap it sits in
    d_start_lookback_s=6.0,  # move a D start back to the first out-of-pitch frame
    w_tag_team=0.1,       # build up / progression team prior (log-odds; r2 0.4, r3 0.1: untestable on the bench)
    w_tag_goal=1.0,       # dead log-odds on the long ball gap nearest a goal tag's end
    goal_win_s=15.0,      # that gap must end within this many seconds of the goal tag's end
    prekick_s=180.0,      # 'kick off (start)' tag within [first-10 s, first+prekick_s]: D before it
    w_gap=3.0,            # dead log-odds on long, slow raw-ball gaps (r1 4; r2 3: the retrained
                          # dead model now sees long gaps itself, bench flat 3-5)
    gap_s=3.0,            # minimum raw-ball gap length for the long-gap term
    gap_speed_ms=1.75,    # speed centre of the long-gap ramp (round 1: hard gate < 2.0 m/s)
    gap_soft=0.2,         # 0: hard gate; > 0: sigmoid width (m/s) of the soft speed / length ramp
    contrast_beta=1.25,   # local-contrast team logit: l + beta * deadzone(l - moving_mean(l, hw), gate)
    contrast_hw=100,      # half width (frames) of that moving mean
    contrast_gate=1.25,   # dead zone (log-odds) of the contrast
    contrast_hs=0,        # optional short pre-smoothing (half width, frames) of l in the contrast
    post_v2=1,            # 1: v2's post chain (gap-end snap both ways, out look-back, merge_short_v2)
    snap_win_s=1.0,       # (post_v2) D end -> nearest long ball-gap end within this window
    out_lookback_s=6.0,   # (post_v2) alive -> D start back to the first out frame in this window
    keep_s=0.5,           # (post_v2) ... staying this long after the possession's start
    merge_v2=1,           # 1: shortest-first, recompute-after-each-merge short-segment merge
    w_out=0.0,            # dead log-odds on ball-out frames (v2 uses 4)
)
_OVERRIDES = sorted(k for k in CFG if "LDE_" + k.upper() in os.environ)
if os.environ.get("LDE_SWEEP") == "1":      # tuning only; ignored unless explicitly enabled
    for _k in _OVERRIDES:
        CFG[_k] = float(os.environ["LDE_" + _k.upper()])
    if _OVERRIDES:
        print("[lde] WARNING: CFG overridden from the environment (LDE_SWEEP=1): "
              + ", ".join("{}={}".format(k, CFG[k]) for k in _OVERRIDES), file=sys.stderr)
elif _OVERRIDES:
    print("[lde] WARNING: LDE_* variables are set but ignored (LDE_SWEEP is not 1): "
          + ", ".join(_OVERRIDES), file=sys.stderr)

_MODELS = {}


def _models():
    if not _MODELS:
        import lightgbm as lgb
        _MODELS["d"] = lgb.Booster(model_file=os.path.join(HERE, "lde_model_d.txt"))
        _MODELS["ha"] = lgb.Booster(model_file=os.path.join(HERE, "lde_model_ha.txt"))
        _MODELS["c"] = lgb.Booster(model_file=os.path.join(HERE, "lde_model_c.txt"))   # mandatory
        with open(os.path.join(HERE, "lde_model_meta.json")) as handle:
            _MODELS["meta"] = json.load(handle)
        # Fail closed: a missing key / file or an old meta must crash, never change the output.
        want = _MODELS["meta"]["lgb_version"]
        assert lgb.__version__ == want, \
            "lightgbm {} installed, models were trained and checked with {}".format(lgb.__version__, want)
        digests = _MODELS["meta"]["sha256"]
        assert set(digests) == {"d", "ha", "c"}, "model_meta.json sha256 must cover d, ha and c"
        import hashlib                     # a retrain must change model_meta.json too (loud)
        for w, digest in sorted(digests.items()):
            with open(os.path.join(HERE, "lde_model_{}.txt".format(w)), "rb") as handle:
                got = hashlib.sha256(handle.read()).hexdigest()
            assert got == digest, "model_{}.txt sha256 {} != model_meta.json {}".format(w, got, digest)
    return _MODELS


def frame_probs(players, slot_team, ball, sign_half1, blocks, pitch, fps, models=None):
    sign = fe.frame_sign(len(ball), sign_half1, blocks)
    F, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
    m = models or _models()
    assert names == m["meta"]["names"], "feature list differs from the trained model"
    p_dead = m["d"].predict(F, num_threads=1)
    perm, signs = fe.swap_index(names, kinds)
    p1 = m["ha"].predict(F, num_threads=1)
    p2 = m["ha"].predict(F[:, perm] * signs, num_threads=1)
    p_home = 0.5 * (p1 + 1.0 - p2)                     # exact H/A symmetry
    aux = dict(F=F, names=names, p_change=None)
    if m.get("c") is not None:
        aux["p_change"] = 0.5 * (m["c"].predict(F, num_threads=1)
                                 + m["c"].predict(F[:, perm] * signs, num_threads=1))
        aux["change_prior"] = m["meta"].get("change_prior", 0.04)
    return p_dead, p_home, aux


def long_gap_gated(players, ball, fps, gap_s, speed_ms, soft=None):
    """Weight in [0, 1] on raw-ball gaps: long gaps with slow tracked players.

    soft 0 (round 1): 1 on gaps longer than gap_s whose mean tracked-player speed is < speed_ms.
    soft s > 0 (round 2): sigmoid((speed_ms - mean_speed) / s) * clip((len - (gap_s - 1)) / 3, 0, 1),
    a ramp instead of the two knife edges (both thresholds were picked on Metrica only)."""
    soft = CFG["gap_soft"] if soft is None else soft
    vel = np.full(players.shape, np.nan, np.float32)
    vel[1:] = (players[1:] - players[:-1]) * fps
    spd = np.sqrt((vel ** 2).sum(-1))
    spd[spd > 12.0] = np.nan
    ms = pp.moving_mean(np.nanmean(spd, 1), fps)
    lg = np.zeros(len(ball))
    for s, e in zip(*pp.runs_of(np.isnan(ball[:, 0]))):
        m = np.nanmean(ms[s:e])
        if soft > 0:
            if not np.isfinite(m):
                m = 0.0
            ramp = np.clip((e - s - (gap_s - 1.0) * fps) / (3.0 * fps), 0.0, 1.0)
            lg[s:e] = ramp / (1.0 + np.exp(-(speed_ms - m) / soft))
        elif e - s > gap_s * fps and m < speed_ms:
            lg[s:e] = 1.0
    return lg


def long_gaps(raw_seen, fps, min_s=3.0):
    return [(int(s), int(e)) for s, e in zip(*pp.runs_of(~raw_seen)) if e - s > min_s * fps]


def tag_priors(tags, names, n, fps, raw_seen=None, goal_win_s=15.0):
    """Team prior from build up / progression tags; dead prior on the stoppage gap of each goal.

    A goal tag runs from the goal to the restart kick-off, so the dead prior is put on the long
    raw-ball gap whose end is nearest the tag's end (within goal_win_s), not on a fixed window
    after the tag (that window is live play after the kick-off)."""
    team = np.zeros(n)
    dead = np.zeros(n)
    home_name, away_name = names[0].strip().lower(), names[1].strip().lower()
    gaps = long_gaps(raw_seen, fps) if raw_seen is not None else []

    def side(t):
        t = (t or "").strip().lower()
        if t == home_name:
            return 1.0
        if t == away_name:
            return -1.0
        return 0.0
    for code, s, e, t, _kind in tags:
        s, e = max(0, s), min(n, max(e, s + 1))
        if code in ("build up", "progression"):
            team[s:e] += side(t)
        elif code == "goals" and gaps:
            g = min(gaps, key=lambda g: abs(g[1] - e))
            if abs(g[1] - e) < goal_win_s * fps:
                dead[g[0]:g[1]] = 1.0
    return np.clip(team, -1, 1), dead


def restart_after(raw_seen, bsm, s, fps, horizon_s=20.0, miss_s=1.0, move_s=0.5, move_m=1.0,
                  fallback_s=5.0):
    """First frame after s where the raw ball reappears after >= miss_s missing, or a seen
    stationary ball starts moving (> move_m over move_s). Fallback s + fallback_s (v2's rule)."""
    n = len(raw_seen)
    miss_k, move_k = int(miss_s * fps), int(move_s * fps)
    for t in range(max(s, 1), min(n - move_k, s + int(horizon_s * fps))):
        if raw_seen[t]:
            if t >= miss_k and not raw_seen[t - miss_k:t].any():
                return t
            if t >= move_k and raw_seen[t - move_k] and raw_seen[t + move_k]:
                if (np.hypot(*(bsm[t] - bsm[t - move_k])) < 0.5
                        and np.hypot(*(bsm[t + move_k] - bsm[t])) > move_m):
                    return t
    return min(n, s + int(fallback_s * fps))


def kickoff_restarts(tags, blocks, ball, fps, prekick_s):
    """{block first frame: restart frame} for a 'kick off (start)' tag near each block start.

    The restart is the first ball reappearance / stationary-ball move after the tag start, kept
    only if it falls inside the tag; otherwise the tag start itself (conservative)."""
    raw_seen = ~np.isnan(ball[:, 0])
    if not tags or not raw_seen.any():
        return {}
    b = ball.astype(np.float64)
    bsm = np.stack([pp.moving_mean(b[:, 0], 2), pp.moving_mean(b[:, 1], 2)], 1)
    bsm[~raw_seen] = np.nan
    out = {}
    for _, first, last in blocks:
        for code, s, e, _t, kind in sorted(tags, key=lambda x: x[1]):
            if code == "set pieces" and "kick off (start)" in str(kind).lower() \
                    and first - 10 * fps <= s < first + prekick_s * fps:
                rs = restart_after(raw_seen, bsm, max(s, 0), fps)
                if rs > max(e, s + 1):       # no clean restart inside the tag: stop at its start
                    rs = s
                if rs > first:
                    out[first] = min(rs, last + 1)
                break
    return out


def viterbi_tv(E, sw, din, dout):
    """3-state Viterbi (H, A, D) with a per-frame H <-> A switch cost sw[t]."""
    T = len(E)
    back = np.zeros((T, 3), np.int8)
    e = E.tolist()
    sw = sw.tolist()
    sH, sA, sD = e[0]
    for t in range(1, T):
        c = sw[t]
        eh, ea, ed = e[t]
        # into H
        a, b, d = sH, sA - c, sD - dout
        if a >= b and a >= d:
            nH, bh = a, 0
        elif b >= d:
            nH, bh = b, 1
        else:
            nH, bh = d, 2
        a, b, d = sH - c, sA, sD - dout
        if b >= a and b >= d:
            nA, ba = b, 1
        elif a >= d:
            nA, ba = a, 0
        else:
            nA, ba = d, 2
        a, b, d = sH - din, sA - din, sD
        if d >= a and d >= b:
            nD, bd = d, 2
        elif a >= b:
            nD, bd = a, 0
        else:
            nD, bd = b, 1
        back[t] = (bh, ba, bd)
        sH, sA, sD = nH + eh, nA + ea, nD + ed
    path = np.empty(T, np.int8)
    path[-1] = int(np.argmax([sH, sA, sD]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def snap_to_peak(path, pc, fps, window_s):
    """Move each H <-> A change to the frame of highest hand-over probability nearby."""
    path = path.copy()
    w = int(window_s * fps)
    segs = pp.segments(path)
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


def snap_dead_edges(path, out, raw_seen, fps, end_s, look_s):
    path = path.copy()
    segs = pp.segments(path)
    for i in range(1, len(segs)):
        s0, _, l0 = segs[i - 1]
        s, e, l = segs[i]
        if l == D and l0 != D and look_s > 0:          # D start back to the first out frame
            lo = max(s0 + fps // 2, s - int(look_s * fps))
            hits = np.where(out[lo:s])[0]
            if len(hits):
                path[lo + int(hits[0]):s] = D
        if l0 == D and l != D and end_s > 0:           # D end forward to the gap's end
            if not raw_seen[s]:
                k = s
                lim = min(e - 1, s + int(end_s * fps))
                while k < lim and not raw_seen[k]:
                    k += 1
                if k < lim:
                    path[s:k] = D
    return path


def snap_gap_ends(path, gaps, fps, win_s):
    """(v2) Each D end -> the end of the nearest long raw-ball gap within win_s, either way (if
    that gap ends after the D spell starts)."""
    path = path.copy()
    if not gaps:
        return path
    w = int(win_s * fps)
    ge = np.array([g[1] for g in gaps])
    for s, e, lab in pp.segments(path):
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


def snap_dead_start_back(path, out, fps, lookback_s, keep_s):
    """(v2) Each alive -> D change: start D at the first ball-out frame in the previous
    lookback_s, staying keep_s after the possession's start."""
    path = path.copy()
    segs = pp.segments(path)
    lb, keep = int(lookback_s * fps), int(keep_s * fps)
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
    """(v2) Shortest offending segment first, segments recomputed after every merge. prev == next
    -> that label; short alive between H and A -> the next team; short alive with exactly one D
    neighbour -> its alive neighbour; D-x-D kept. Order-independent, unlike pp.merge_short."""
    min_len = {H: int(P["min_poss_s"] * fps), A: int(P["min_poss_s"] * fps),
               D: int(P["min_dead_s"] * fps)}
    path = path.copy()
    while True:
        segs = pp.segments(path)
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
            else:
                path[s:e] = nxt if prev == D else prev
            done = True
            break
        if not done:
            break
    return path


def decode(p_dead, p_home, aux, blocks, n, fps, tags=(), names=("Home", "Away"), cfg=None,
           shape_logit=None):
    c = dict(CFG, **(cfg or {}))
    tau = c["tau"]
    pdd = np.clip(p_dead, 1e-3, 1 - 1e-3)
    ph = np.clip(p_home, 1e-3, 1 - 1e-3)
    lh = np.log(ph / (1 - ph))
    if c["contrast_beta"] > 0:                    # local contrast: lets short opposing dips cross 0
        q = np.clip(p_home, 1e-6, 1 - 1e-6)
        l6 = np.log(q / (1 - q))
        ls = pp.moving_mean(l6, int(c["contrast_hs"])) if c["contrast_hs"] > 0 else l6
        cc = ls - pp.moving_mean(l6, int(c["contrast_hw"]))
        cc = np.sign(cc) * np.maximum(np.abs(cc) - c["contrast_gate"], 0.0)
        lh = np.clip(l6 + c["contrast_beta"] * cc, -np.log(999.0), np.log(999.0))
    if shape_logit is not None and c["beta"] > 0:
        lh = (1 - c["beta"]) * lh + c["beta"] * shape_logit
    ld = np.log(pdd / (1 - pdd))
    if aux.get("long_gap") is not None and c["w_gap"] != 0:
        ld = ld + c["w_gap"] * aux["long_gap"]
    F, nm = aux["F"], aux["names"]
    col = {k: i for i, k in enumerate(nm)}
    raw_seen = F[:, col["ball_seen"]] > 0.5
    out = np.nan_to_num(F[:, col["ball_out"]], nan=0.0) > 0.5
    out = pp.moving_mean(out.astype(float), 3) > 0.5
    if c["w_out"] != 0:
        ld = ld + c["w_out"] * out
    if len(tags):
        tt, td = tag_priors(tags, names, n, fps, raw_seen, c["goal_win_s"])
        lh = lh + c["w_tag_team"] * tt
        ld = ld + c["w_tag_goal"] * td
    ph = 1 / (1 + np.exp(-lh))
    pdd = 1 / (1 + np.exp(-ld))
    ph = np.clip(ph, 1e-3, 1 - 1e-3)
    pdd = np.clip(pdd, 1e-3, 1 - 1e-3)
    E = tau * np.stack([np.log(ph) + np.log(1 - pdd), np.log(1 - ph) + np.log(1 - pdd),
                        np.log(pdd)], 1)
    scale = tau / 0.15
    P = dict(pp.PARAMS, switch_cost=c["switch_cost"] * scale,
             dead_in_cost=c["dead_cost"] * scale, dead_out_cost=c["dead_cost"] * scale,
             min_poss_s=c["min_poss_s"], min_dead_s=c["min_dead_s"],
             refine_window_s=c["refine_window_s"], control_radius_m=c["control_radius_m"])
    d_home = F[:, col["H_d1"]].astype(np.float64)
    d_away = F[:, col["A_d1"]].astype(np.float64)
    seen = ~np.isnan(d_home) | ~np.isnan(d_away)
    gaps_all = long_gaps(raw_seen, fps, c["gap_s"])
    pc = aux.get("p_change")
    use_c = pc is not None and c["gamma"] > 0
    if use_c:
        pcc = np.clip(pc, 1e-3, 1 - 1e-3)
        prior = aux["change_prior"]
        bonus = np.log(pcc / (1 - pcc)) - np.log(prior / (1 - prior))
        sw = np.maximum(0.02, P["switch_cost"] - c["gamma"] * tau * bonus)
    pred = np.full(n, 3, np.int8)
    for _, first, last in blocks:
        seg = slice(first, last + 1)
        if use_c:
            path = viterbi_tv(E[seg], sw[seg], P["dead_in_cost"], P["dead_out_cost"])
        else:
            path = pp.viterbi(E[seg], P)
        if c["post_v2"]:
            g = [(s0 - first, e0 - first) for s0, e0 in gaps_all if s0 >= first and e0 <= last + 1]
            path = snap_gap_ends(path, g, fps, c["snap_win_s"])
            path = snap_dead_start_back(path, out[seg], fps, c["out_lookback_s"], c["keep_s"])
        if c["merge_v2"] or c["post_v2"]:
            path = merge_short_v2(path, P, fps)
        else:
            path = pp.merge_short(path, P)
        if pc is not None and c["snap_mode"] >= 0.5:
            path = snap_to_peak(path, pc[seg], fps, c["refine_window_s"])
        else:
            path = pp.snap_boundaries(path, dict(d_home=d_home[seg], d_away=d_away[seg],
                                                 out=out[seg], seen=seen[seg]), P)
        if not c["post_v2"]:
            path = snap_dead_edges(path, out[seg], raw_seen[seg], fps, c["d_end_snap_s"],
                                   c["d_start_lookback_s"])
        rs = aux.get("kick_restarts", {}).get(first)
        if rs is not None:                              # ball dead before the half's kick-off
            path[:rs - first] = D
        pred[seg] = path
    return pred


def v1_shape_logit(players, slot_team, ball, sign_half1, blocks, tags, names, pitch):
    """v1's in-match pseudo-label team-shape logit (self-generated labels), for the beta blend."""
    P = dict(pp.PARAMS, temperature=1.0, w_ball_near=0.0, w_ball_far=0.0, w_tag=0.0,
             w_shape=1.0)
    E, _ = pp.build_evidence(players, ball, slot_team, sign_half1, blocks, [], names, pitch, P,
                             lambda *a, **k: None)
    return np.clip(E[:, 0] - E[:, 1], -3, 3)


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    p_dead, p_home, aux = frame_probs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    aux["long_gap"] = long_gap_gated(players, ball, fps, CFG["gap_s"], CFG["gap_speed_ms"])
    aux["kick_restarts"] = kickoff_restarts(tags, blocks, ball, fps, CFG["prekick_s"])
    shape = None
    if CFG["beta"] > 0:
        shape = v1_shape_logit(players, slot_team, ball, sign_half1, blocks, tags, names, pitch)
    return decode(p_dead, p_home, aux, blocks, len(ball), fps, tags, names, shape_logit=shape)


def evidence_probs(p_dead, p_home, aux, n, fps, tags=(), names=("Home", "Away"), cfg=None):
    """(ballfree_dead addition) The per-frame P(home | alive) and P(dead) that decode() turns
    into Viterbi emissions (same contrast, long-gap, out and tag terms), for predict_proba."""
    c = dict(CFG, **(cfg or {}))
    pdd = np.clip(p_dead, 1e-3, 1 - 1e-3)
    ph = np.clip(p_home, 1e-3, 1 - 1e-3)
    lh = np.log(ph / (1 - ph))
    if c["contrast_beta"] > 0:
        q = np.clip(p_home, 1e-6, 1 - 1e-6)
        l6 = np.log(q / (1 - q))
        ls = pp.moving_mean(l6, int(c["contrast_hs"])) if c["contrast_hs"] > 0 else l6
        cc = ls - pp.moving_mean(l6, int(c["contrast_hw"]))
        cc = np.sign(cc) * np.maximum(np.abs(cc) - c["contrast_gate"], 0.0)
        lh = np.clip(l6 + c["contrast_beta"] * cc, -np.log(999.0), np.log(999.0))
    ld = np.log(pdd / (1 - pdd))
    if aux.get("long_gap") is not None and c["w_gap"] != 0:
        ld = ld + c["w_gap"] * aux["long_gap"]
    F, nm = aux["F"], aux["names"]
    col = {k: i for i, k in enumerate(nm)}
    raw_seen = F[:, col["ball_seen"]] > 0.5
    out = np.nan_to_num(F[:, col["ball_out"]], nan=0.0) > 0.5
    out = pp.moving_mean(out.astype(float), 3) > 0.5
    if c["w_out"] != 0:
        ld = ld + c["w_out"] * out
    if len(tags):
        tt, td = tag_priors(tags, names, n, fps, raw_seen, c["goal_win_s"])
        lh = lh + c["w_tag_team"] * tt
        ld = ld + c["w_tag_goal"] * td
    ph = np.clip(1 / (1 + np.exp(-lh)), 1e-3, 1 - 1e-3)
    pdd = np.clip(1 / (1 + np.exp(-ld)), 1e-3, 1 - 1e-3)
    return ph, pdd
