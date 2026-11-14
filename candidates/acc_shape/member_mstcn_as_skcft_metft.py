"""MS-TCN member of v11: the m5c_C192ls -> SkillCorner fine-tune -> Metrica fine-tune chain (the recipes of
nn_pff/pipeline_c192ls_skcft_metft.py, v9's MS-TCN) retrained on 223 inputs (repo features + 29 ACC + SHAPE columns),
weights mstcn_as_skcft_metft.npz; nn_pff/pipeline.py unchanged (network, decoder), only features and weights differ."""
import _common

_mod = _common.mstcn()
predict = _mod.predict
predict_proba = _mod.predict_proba
