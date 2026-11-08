"""temporal_tcn_v2, training-target variant 'ctrl' (bench comparison only; pipeline.py ships the winner).

pipeline.py executed under a private module name with its weights pointed at tcn_weights_ctrl.npz
(loaded, like the shipped file, through dfl_folds.resolve). See README.md.
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("temporal_tcn_v2_variant_ctrl", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.WEIGHTS = os.path.join(HERE, "tcn_weights_ctrl.npz")

predict = _base.predict
predict_proba = _base.predict_proba
