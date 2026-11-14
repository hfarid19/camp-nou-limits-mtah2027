"""MS-TCN member of v12: the m5c_C192ls -> SkillCorner fine-tune -> Metrica fine-tune chain (v11's recipes) retrained on the
v12 MS-TCN inputs (repo features + _common.MS_GROUPS blocks, fe_v12), weights mstcn_v12_skcft_metft.npz."""
import _common

_mod = _common.mstcn()
predict = _mod.predict
predict_proba = _mod.predict_proba
