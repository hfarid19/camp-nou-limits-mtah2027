"""Bench wrapper for a trained configuration (experiments only; pipeline.py ships the chosen one).

pipeline.py executed under a private module name with its weights pointed at $NN_PFF_MODEL (a
config's full.npz under $MTAH_WORK/arch/nn_pff/<config>; several files separated by ':' = a seed
ensemble whose softmaxes are averaged). The weights still load through
pff_folds.resolve, so bench --dfl2 / run_pff.py swap in the config's fold models (the folds JSON
maps the full model's sha256 to the config folder).
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(os.path.basename(HERE) + "_cfg_variant", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_attr = "MODEL_PATH" if hasattr(_base, "MODEL_PATH") else "WEIGHTS"
_m = [os.path.abspath(p) for p in os.environ["NN_PFF_MODEL"].split(":")]     # a:b:c = seed ensemble
setattr(_base, _attr, _m[0] if len(_m) == 1 else _m)

predict = _base.predict
predict_proba = _base.predict_proba
