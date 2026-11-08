"""ball_radar_v3 (ball_radar retrained with the attack-direction fix, see README.md): Ball Radar-style ball / possession inference (set attention + hierarchical Bi-GRU)
fused with learned_dfl_evidence's LightGBM frame evidence, then LDE's Viterbi decoder.

See README.md for the method, references, data sources, variants and scores.

predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps) -> codes
predict_proba(same)  -> (N, 3) P(H), P(A), P(D): the fused per-frame evidence the decoder uses.
"""

import hashlib
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))      # candidates/ (dfl_folds)
sys.path.insert(0, HERE)
import dfl_folds                   # noqa: E402
import features as fe              # noqa: E402
import features_br as fb           # noqa: E402
import lde_core as lde             # noqa: E402
import radar_common as rc          # noqa: E402

warnings.filterwarnings("ignore")

RADAR_PATH = os.path.join(HERE, "radar.npz")
CFG = dict(
    # final: the stacked models alone (LDE-free). The other weights reproduce the variants in
    # README.md (they need learned_dfl_evidence's model files next to lde_core.py).
    a_h=0.0,        # weight of LDE's own team logit (LDE model_ha)
    a_d=0.0,        # weight of LDE's own dead logit (LDE model_d)
    w_h=0.0,        # weight of the radar team head log-odds log P(H) - log P(A)
    w_d=0.0,        # weight of the radar dead-head log-odds
    b_h=1.0,        # weight of the stacked team model (features on the imputed ball + radar)
    b_d=1.0,        # weight of the stacked dead model
    br_c=1.0,       # 1: hand-over snapping uses the stacked hand-over model (model_brc)
    br_feats=1.0,   # 1: stacked features on the imputed ball + radar columns; 0: plain LDE
                    # features (ablation only)
)
if os.environ.get("BR_SWEEP") == "1":          # tuning only (DFL LOGO); ignored otherwise
    for _k in CFG:
        if "BR_" + _k.upper() in os.environ:
            CFG[_k] = float(os.environ["BR_" + _k.upper()])
    for _k in lde.CFG:
        if "BR_LDE_" + _k.upper() in os.environ:
            lde.CFG[_k] = float(os.environ["BR_LDE_" + _k.upper()])
    print("[ball_radar] sweep CFG", CFG, file=sys.stderr)

_CACHE = {}


def _key(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def radar(players, slot_team, ball, sign_half1, blocks, fps):
    """Ball Radar outputs at every frame of the graded blocks (linear interpolation of the 5 Hz
    network outputs; home/away-symmetrised by averaging with the swapped run).
    -> dict(team (N, 3) P(H, A, D), ball (N, 2) metres (NaN outside blocks), g_dead, poss_h,
    poss_a (player-level possession mass per team))."""
    wpath = dfl_folds.resolve(RADAR_PATH)
    cdir = os.environ.get("BR_RCACHE") if os.environ.get("BR_SWEEP") == "1" else None
    if cdir:                                    # sweep-only disk cache (deterministic outputs)
        with open(wpath, "rb") as handle:
            wsha = hashlib.sha1(handle.read()).hexdigest()
        cpath = os.path.join(cdir, "r_" + _key(players, ball, np.asarray(blocks)) + wsha[:12] + ".npz")
        if os.path.isfile(cpath):
            z = np.load(cpath)
            return {k: z[k] for k in z.files}
    if os.environ.get("BENCH_DFL_FOLD") is None:      # fail closed on a changed radar file
        with open(os.path.join(HERE, "model_br_meta.json")) as handle:
            want = json.load(handle)["radar_sha256"]
        with open(wpath, "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == want, "radar.npz differs from model_br_meta.json"
    out = rc.run(rc.load_weights(wpath), players, slot_team, ball, sign_half1, blocks, fps)
    if cdir:
        os.makedirs(cdir, exist_ok=True)
        np.savez(cpath, **out)
    return out


_BR = {}


def stacked(players, slot_team, ball, sign_half1, blocks, pitch, fps, r):
    """Stacked LightGBM (model_brd / model_brha): LDE features on the radar-imputed ball plus
    radar evidence columns. -> p_dead, p_home (home/away-symmetrised)."""
    if not _BR:
        import lightgbm as lgb
        with open(os.path.join(HERE, "model_br_meta.json")) as handle:
            _BR["meta"] = json.load(handle)
        assert lgb.__version__ == _BR["meta"]["lgb_version"], "lightgbm version differs from training"
        for w in ("d", "ha", "c") if CFG["br_c"] else ("d", "ha"):
            path = os.path.join(HERE, "model_br{}.txt".format(w))
            with open(path, "rb") as handle:
                assert hashlib.sha256(handle.read()).hexdigest() == _BR["meta"]["models"][w]["sha256"]
            _BR[w] = lgb.Booster(model_file=path)
    sf = fb.frame_sign(len(ball), sign_half1, blocks)
    if CFG["br_feats"]:
        F, names, kinds = fb.compute(players, slot_team, ball, sf, pitch, fps, ball_imp=r["ball"], radar=r)
    else:
        F, names, kinds = fb.compute(players, slot_team, ball, sf, pitch, fps)
    assert names == _BR["meta"]["names"]
    perm, signs = fb.swap_index(names, kinds)
    pd = _BR["d"].predict(F, num_threads=1)
    p1 = _BR["ha"].predict(F, num_threads=1)
    p2 = _BR["ha"].predict(F[:, perm] * signs, num_threads=1)
    pc = None
    if CFG["br_c"]:
        pc = 0.5 * (_BR["c"].predict(F, num_threads=1) + _BR["c"].predict(F[:, perm] * signs, num_threads=1))
    return pd, 0.5 * (p1 + 1.0 - p2), pc, F, names


def raw_ball_out(ball, pitch, fps):
    """features.compute's ball_out column on the raw ball (short gaps filled, NaN in longer
    gaps), as LDE's decoder uses it; the stacked features hold it on the imputed ball."""
    gapf = int(round(fe.GAP_FILL_S * fps))
    bx = fe.fill_short_gaps(ball[:, 0].astype(np.float64), gapf)
    by = fe.fill_short_gaps(ball[:, 1].astype(np.float64), gapf)
    L, Wd = pitch
    ox = np.maximum(np.maximum(-bx, bx - L), 0.0)
    oy = np.maximum(np.maximum(-by, by - Wd), 0.0)
    return np.sqrt(ox ** 2 + oy ** 2).astype(np.float32)


def evidence(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    k = _key(players, ball, np.asarray(slot_team), np.asarray(blocks), np.float64(sign_half1))
    if k in _CACHE:
        return _CACHE[k]
    lde_free = CFG["a_h"] == 0 and CFG["a_d"] == 0 and CFG["br_c"]
    if lde_free:          # no LDE booster at all: the decoder only needs these aux entries
        p_dead = p_home = np.full(len(ball), 0.5)
        aux = {}
    else:
        p_dead, p_home, aux = lde.frame_probs(players, slot_team, ball, sign_half1, blocks, pitch, fps)
    aux["long_gap"] = lde.long_gap_gated(players, ball, fps, lde.CFG["gap_s"], lde.CFG["gap_speed_ms"])
    aux["kick_restarts"] = lde.kickoff_restarts(tags, blocks, ball, fps, lde.CFG["prekick_s"])
    r = radar(players, slot_team, ball, sign_half1, blocks, fps)
    eps = 1e-4
    bh = bd = 0.0
    if CFG["b_h"] != 0 or CFG["b_d"] != 0:
        sbd, sbh, sbc, F, fnames = stacked(players, slot_team, ball, sign_half1, blocks, pitch, fps, r)
        if sbc is not None:
            aux["p_change"] = sbc
        if lde_free:
            F = F.copy()
            F[:, fnames.index("ball_out")] = raw_ball_out(ball, pitch, fps)
            aux.update(F=F, names=fnames, change_prior=0.04)
        sbd, sbh = np.clip(sbd, eps, 1 - eps), np.clip(sbh, eps, 1 - eps)
        bh, bd = np.log(sbh / (1 - sbh)), np.log(sbd / (1 - sbd))
    ph = np.clip(p_home, eps, 1 - eps)
    pd = np.clip(p_dead, eps, 1 - eps)
    rt = np.clip(r["team"], eps, 1.0)
    lh = (CFG["a_h"] * np.log(ph / (1 - ph)) + CFG["w_h"] * (np.log(rt[:, 0]) - np.log(rt[:, 1]))
          + CFG["b_h"] * bh)
    ld = (CFG["a_d"] * np.log(pd / (1 - pd)) + CFG["w_d"] * (np.log(rt[:, 2]) - np.log(1 - rt[:, 2] + eps))
          + CFG["b_d"] * bd)
    ph2 = 1 / (1 + np.exp(-lh))
    pd2 = 1 / (1 + np.exp(-ld))
    _CACHE[k] = (pd2, ph2, aux, r)
    return _CACHE[k]


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    pd, ph, aux, _ = evidence(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)
    return lde.decode(pd, ph, aux, blocks, len(ball), fps, tags, names)


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    pd, ph, _, _ = evidence(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps)
    p = np.stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd], 1)
    return p / p.sum(1, keepdims=True)
