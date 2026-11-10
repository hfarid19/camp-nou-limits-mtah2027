"""lsw TCN member (o_ls05, tcn_ls.npz) with test-time averaging over 4 jittered input copies and per-game re-normalisation all_a50 (renorm_members.tcn_tta)."""
import renorm_members as _rm

_mod = _rm.tcn_tta(4, "all_a50")
predict = _mod.predict
predict_proba = _mod.predict_proba
