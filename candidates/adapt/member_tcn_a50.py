"""lsw TCN member (o_ls05, tcn_ls.npz) with per-game re-normalisation all_a50 (renorm_members.py)."""
import renorm_members as _rm

_mod = _rm.tcn("all_a50")
predict = _mod.predict
predict_proba = _mod.predict_proba
