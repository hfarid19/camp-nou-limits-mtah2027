"""handover_mstcn_v2, training-target variant 'raw' (bench comparison only; pipeline.py ships the winner).

pipeline.py executed under a private module name with its weights pointed at model_tcn_raw.npz
(loaded, like the shipped file, through dfl_folds.resolve). See README.md.
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("handover_mstcn_v2_variant_raw", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.MODEL_PATH = os.path.join(HERE, "model_tcn_raw.npz")

predict = _base.predict
predict_proba = _base.predict_proba
