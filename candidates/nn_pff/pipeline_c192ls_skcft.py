"""nn_pff, "C192ls_skcft" variant (round 5, "+SKC" experiment, SKC-FT): m5c_C192ls (pipeline_c192ls.py) fine-tuned
on the 7 DFL + 48 PFF games + the 20 SkillCorner A-League open-data games (lr 5e-5 = 0.1 x the parent's, 5 epochs =
750 iterations at batch 8, Adam at constant lr, label smoothing 0.05, the parent's feature normalisation kept);
weights mstcn_c192ls_skcft.npz. Same network, features and decoder as the parent: only the weights differ.
SkillCorner open data: MIT licence, (c) 2020 SkillCorner; notice in TECHNICAL.md, section 4 ("Third-party data notices").

pipeline.py executed under a private module name with its weights pointed at mstcn_c192ls_skcft.npz (still
loaded through pff_folds.resolve, so the DFL bench uses that file's leave-one-game-out fold models).
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nn_pff_variant_c192ls_skcft", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
setattr(_base, "MODEL_PATH" if hasattr(_base, "MODEL_PATH") else "WEIGHTS", os.path.join(HERE, "mstcn_c192ls_skcft.npz"))

predict = _base.predict
predict_proba = _base.predict_proba
