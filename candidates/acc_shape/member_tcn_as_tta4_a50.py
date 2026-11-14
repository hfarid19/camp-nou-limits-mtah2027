"""TCN member of v11: the o_ls05 recipe retrained on 223 inputs (repo features + 29 ACC + SHAPE columns), weights tcn_as.npz,
with test-time averaging over 4 jittered input copies and per-game re-normalisation all_a50 (as adapt/member_tcn_tta4_a50.py)."""
import _common

_mod = _common.tcn_tta(4, "all_a50")
predict = _mod.predict
predict_proba = _mod.predict_proba
