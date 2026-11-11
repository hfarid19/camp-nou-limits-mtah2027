"""nn_pff, "C192ls_skcft_metft" variant (round 5, skcmetft experiment, prereg addenda 2026-11-11 12:31 / 12:46 CST):
m5c_C192ls_skcft (pipeline_c192ls_skcft.py, the MS-TCN of submission v8) fine-tuned once more with Metrica's three open
sample games added (the 10 Nov MET-FT recipe applied to the shipped skcft: 750 iterations at batch 8, crop 1024, Adam
at constant lr 5e-5, no clip / EMA, seed 0, label smoothing 0.05; each batch mixes 1:1 crops of the parent's own
training data (7 DFL + 48 PFF + 20 SkillCorner games, drawn with the parent's sampling weights) and crops of Metrica
Sample Games 1-3 (strict Metrica-rule labels); the parent's feature normalisation kept); weights
mstcn_c192ls_skcft_metft.npz (sha256 a7a242c352e3...). Same network, features and decoder as the parent: only the
weights differ.
Training data notices: SkillCorner open data, MIT licence, (c) 2020 SkillCorner (notice in TECHNICAL.md, "Data
Sources"); Metrica Sports sample data, github.com/metrica-sports/sample-data (no licence file; its README asks users to
be responsible with the data and to acknowledge the source in any public use).

pipeline.py executed under a private module name with its weights pointed at mstcn_c192ls_skcft_metft.npz (still
loaded through pff_folds.resolve -> dfl_folds.resolve, so the DFL bench uses that file's leave-one-game-out fold models
and refuses to score DFL games when the file's sha256 has no fold mapping in --folds).
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nn_pff_variant_c192ls_skcft_metft", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
setattr(_base, "MODEL_PATH" if hasattr(_base, "MODEL_PATH") else "WEIGHTS", os.path.join(HERE, "mstcn_c192ls_skcft_metft.npz"))

predict = _base.predict
predict_proba = _base.predict_proba
