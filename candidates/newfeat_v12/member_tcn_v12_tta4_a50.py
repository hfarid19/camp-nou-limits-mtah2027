"""TCN member of v12: the o_ls05 recipe retrained on 281 inputs (repo features + ACC + SHAPE + REL + KIN2 + INT columns,
fe_v12), weights tcn_v12.npz, with test-time averaging over 4 jittered input copies and per-game re-normalisation all_a50
(as acc_shape/member_tcn_as_tta4_a50.py)."""
import _common

_mod = _common.tcn_tta(4, "all_a50")
predict = _mod.predict
predict_proba = _mod.predict_proba
