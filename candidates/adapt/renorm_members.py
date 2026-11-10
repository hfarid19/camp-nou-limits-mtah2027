"""Live (shippable) members with per-game feature re-normalisation (renorm.py): the lsw NN members
(nn_pff_tcn/pipeline.py on tcn_ls.npz, nn_pff/pipeline.py on mstcn_w.npz) with the training mean / sd
moved towards the game's own mirror-pooled feature statistics (renorm.VARIANTS[variant]).

The base pipelines are executed unchanged under private names (members.load isolation); only the
step that normalises the features is wrapped:
  TCN     its tcn_probs(F, names, kinds, blocks, W) gets W with mu / sd replaced (the game stats come
          from the same F and blocks, so the evidence cache, decoder and fold lookup are untouched);
  MS-TCN  tcn_outputs is the base function with the model's mean / std replaced after loading.
Deterministic numpy; the game statistics depend only on the positions passed to predict (and the
blocks), never on labels. Weights still go through pff_folds.resolve (DFL LOGO folds on the bench).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.dirname(HERE)
if os.path.join(CAND, "ensemble_ctrl_v6") not in sys.path:
    sys.path.insert(0, os.path.join(CAND, "ensemble_ctrl_v6"))
import members  # noqa: E402
import renorm   # noqa: E402


def tcn(variant, weights="tcn_ls.npz"):
    mod = members.load("nn_pff_tcn", "pipeline.py")
    mod.WEIGHTS = os.path.join(CAND, "nn_pff_tcn", weights)
    base = mod.tcn_probs

    def tcn_probs(F, names, kinds, blocks, W):
        perm, signs = mod.fe.swap_index(names, kinds)
        mom = renorm.game_moments(F, blocks, perm, signs)
        mu, sd, _ = renorm.variant_stats(variant, W["mu"], W["sd"], mom, [str(s) for s in names], perm)
        W2 = dict(W)
        W2["mu"], W2["sd"] = mu.astype(W["mu"].dtype), sd.astype(W["sd"].dtype)
        return base(F, names, kinds, blocks, W2)
    mod._base_tcn_probs = base
    mod.tcn_probs = tcn_probs
    return mod


def mstcn(variant, weights="mstcn_w.npz"):
    mod = members.load("nn_pff", "pipeline.py")
    mod.MODEL_PATH = os.path.join(CAND, "nn_pff", weights)

    def tcn_outputs(players, slot_team, ball, sign_half1, blocks, pitch, fps):
        path = mod.pff_folds.resolve(mod.MODEL_PATH)
        key = mod._key(players, ball, blocks, path) + variant
        if key in mod._CACHE:
            return mod._CACHE[key]
        model = mod.tcn_np.Model(path)
        sign = mod.fe.frame_sign(len(ball), sign_half1, blocks)
        F, names, kinds = mod.fe.compute(players, slot_team, ball, sign, pitch, fps)
        assert names == model.names, "feature list differs from the trained model"
        perm, signs = mod.fe.swap_index(names, kinds)
        mom = renorm.game_moments(F, blocks, perm, signs)
        for s in model.members:
            s.mean, s.std, _ = renorm.variant_stats(variant, s.mean, s.std, mom, list(names), perm)
        n = len(ball)
        pc = np.full((n, 3), 1.0 / 3.0)
        pb = np.zeros((n, 2))
        for _, first, last in blocks:
            for r in range(mod.STRIDE):
                fr = np.arange(first + r, last + 1, mod.STRIDE)
                if len(fr) == 0:
                    continue
                pc[fr], pb[fr] = model.predict(F[fr].astype(np.float64), perm, signs)
        mod._CACHE.clear()
        mod._CACHE[key] = (pc, pb, F, names)
        return mod._CACHE[key]
    mod.tcn_outputs = tcn_outputs
    return mod


def tcn_tta(k=4, variant="base", weights="tcn_ls.npz"):
    """TCN member with test-time averaging over k jittered copies of the input (tta.jitter: smooth AR(1)
    offsets, players 0.5 m / ball 0.3 m, numpy default_rng(1000 + r)) + the original: class and hand-over
    probabilities averaged over the k + 1 passes, then the member's own decoder on the original features.
    variant: optional re-normalisation (each pass uses the ORIGINAL pass's game statistics)."""
    import tta
    mod = tcn(variant, weights) if variant != "base" else members.load("nn_pff_tcn", "pipeline.py")
    if variant == "base":
        mod.WEIGHTS = os.path.join(CAND, "nn_pff_tcn", weights)
    fe = mod.fe

    def evidence(players, slot_team, ball, sign_half1, blocks, pitch, fps):
        key = mod._key(players, ball, blocks) + "tta{}{}".format(k, variant)
        if key not in mod._CACHE:
            mod._CACHE.clear()
            sign = fe.frame_sign(len(ball), sign_half1, blocks)
            F0, names, kinds = fe.compute(players, slot_team, ball, sign, pitch, fps)
            Fs = [F0]
            for r in range(k):
                pj, bj = tta.jitter(players, ball, r, fps)
                Fs.append(fe.compute(pj, slot_team, bj, sign, pitch, fps)[0])
            Ws = mod._weights()
            if variant != "base":
                perm, signs = fe.swap_index(names, kinds)
                mom = renorm.game_moments(F0, blocks, perm, signs)
                W2s = []
                for W in Ws:
                    mu, sd, _ = renorm.variant_stats(variant, W["mu"], W["sd"], mom, [str(s) for s in names], perm)
                    W2 = dict(W)
                    W2["mu"], W2["sd"] = mu.astype(W["mu"].dtype), sd.astype(W["sd"].dtype)
                    W2s.append(W2)
                Ws = W2s
            base_probs = getattr(mod, "_base_tcn_probs", mod.tcn_probs)   # unwrapped: stats set above
            outs = [base_probs(F, names, kinds, blocks, W) for W in Ws for F in Fs]
            P = np.mean([o[0] for o in outs], 0)
            pc = np.mean([o[1] for o in outs], 0)
            mod._CACHE[key] = (P, pc, F0, names, float(np.mean([float(W["change_prior"]) for W in Ws])))
        return mod._CACHE[key]
    mod.evidence = evidence
    return mod
