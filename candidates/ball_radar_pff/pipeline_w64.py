"""ball_radar_pff, round-4 "64" variant: the same recipe retrained on 7 DFL + all 64 PFF games (the 16
spent holdout games included, NN_PFF_ALLOW_HOLDOUT=1), weights in w64/ (radar.npz, model_br{d,ha,c}.txt,
model_br_meta.json); see README.md "Round 4".

pipeline.py executed under a private module name with its weight folder pointed at w64/ (still loaded
through pff_folds.resolve, so bench --dfl2 uses that file's DFL fold models). pipeline.py is unchanged.
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.environ.get("BRP_WEIGHTS") or os.path.join(HERE, "w64")
_spec = importlib.util.spec_from_file_location("ball_radar_pff_variant_w64", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.HERE = WDIR                                   # model_br_meta.json / model_br*.txt
_base.RADAR_PATH = os.path.join(WDIR, "radar.npz")

predict = _base.predict
predict_proba = _base.predict_proba
