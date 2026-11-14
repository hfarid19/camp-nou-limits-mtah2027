"""Shared loader code of the acc_shape members: the base pipelines (nn_pff_tcn/pipeline.py, nn_pff/pipeline.py) executed
unchanged and in isolation (ensemble_ctrl_v6/members.load), with their `fe` module replaced by fe_as (repo features +
29 ACC + SHAPE columns) and their weights pointed at this folder's files. Weights still load through pff_folds.resolve
(-> dfl_folds.resolve), so the DFL bench uses the files' leave-one-game-out fold models and refuses to score DFL games
when a file's sha256 has no fold mapping in --folds."""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
TCN_WEIGHTS = os.path.join(HERE, "tcn_as.npz")
MSTCN_WEIGHTS = os.path.join(HERE, "mstcn_as_skcft_metft.npz")


def fe():
    name = "mtah_acc_shape_fe"
    if name not in sys.modules:
        sp = importlib.util.spec_from_file_location(name, os.path.join(HERE, "fe_as.py"))
        mod = importlib.util.module_from_spec(sp)
        sys.modules[name] = mod
        sp.loader.exec_module(mod)
    return sys.modules[name]


def _adapt():
    """candidates/adapt's renorm_members, renorm, tta (plain imports with that folder on sys.path, then path restored)."""
    saved = list(sys.path)
    sys.path[:0] = [os.path.join(CAND, "adapt"), os.path.join(CAND, "ensemble_ctrl_v6")]
    try:
        import renorm_members
        import renorm
        import tta
    finally:
        sys.path[:] = saved
    return renorm_members, renorm, tta


def tcn_plain(weights=TCN_WEIGHTS):
    rm, _, _ = _adapt()
    mod = rm.members.load("nn_pff_tcn", "pipeline.py")
    mod.WEIGHTS = weights
    mod.fe = fe()
    return mod


def tcn_tta(k=4, variant="all_a50", weights=TCN_WEIGHTS):
    """adapt/renorm_members.tcn_tta(k, variant) with the feature module replaced: renorm_members.tcn (the re-normalised
    TCN, unchanged) and then tcn_tta's evidence() body, copied, computing every pass with fe_as."""
    rm, renorm, tta = _adapt()
    mod = rm.tcn(variant, weights)            # os.path.join(<nn_pff_tcn>, <absolute path>) = the absolute path
    assert mod.WEIGHTS == weights
    f = fe()
    mod.fe = f

    def evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps):
        key = mod._key(players, ball, blocks) + "tta{}{}".format(k, variant)
        if key not in mod._CACHE:
            mod._CACHE.clear()
            sign = f.frame_sign(len(ball), sign_half1, blocks)
            F0, names, kinds = f.compute(players, slot_team, ball, sign, pitch, fps)
            Fs = [F0]
            for r in range(k):
                pj, bj = tta.jitter(players, ball, r, fps)
                Fs.append(f.compute(pj, slot_team, bj, sign, pitch, fps)[0])
            Ws = mod._weights()
            perm, signs = f.swap_index(names, kinds)
            mom = renorm.game_moments(F0, blocks, perm, signs)
            W2s = []
            for W in Ws:
                mu, sd, _ = renorm.variant_stats(variant, W["mu"], W["sd"], mom, [str(s) for s in names], perm)
                W2 = dict(W)
                W2["mu"], W2["sd"] = mu.astype(W["mu"].dtype), sd.astype(W["sd"].dtype)
                W2s.append(W2)
            outs = [mod._base_tcn_probs(F, names, kinds, blocks, W) for W in W2s for F in Fs]
            P = np.mean([o[0] for o in outs], 0)
            pc = np.mean([o[1] for o in outs], 0)
            mod._CACHE[key] = (P, pc, F0, names, float(np.mean([float(W["change_prior"]) for W in Ws])))
        return mod._CACHE[key]
    mod.evidence = evidence
    return mod


def mstcn(weights=MSTCN_WEIGHTS):
    rm, _, _ = _adapt()
    mod = rm.members.load("nn_pff", "pipeline.py")
    mod.MODEL_PATH = weights
    mod.fe = fe()
    return mod
