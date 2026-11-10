"""lsw MS-TCN member (m5c_W, mstcn_w.npz) with per-game re-normalisation all_a50 (renorm_members.py)."""
import renorm_members as _rm

_mod = _rm.mstcn("all_a50")
predict = _mod.predict
predict_proba = _mod.predict_proba
