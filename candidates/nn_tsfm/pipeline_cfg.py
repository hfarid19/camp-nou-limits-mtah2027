"""Bench wrapper for a trained configuration (experiments only; pipeline.py ships the chosen one).

pipeline.py executed under a private module name with WEIGHTS pointed at $NN_TSFM_MODEL (a config's
full.npz under $MTAH_WORK/arch/nn_tsfm/<config>). The weights still load through pff_folds.resolve,
so bench --dfl2 / run_pff.py swap in the config's fold models (the folds JSON maps the full model's
sha256 to the config folder).
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(os.path.basename(HERE) + "_cfg_variant", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.WEIGHTS = os.path.abspath(os.environ["NN_TSFM_MODEL"])

predict = _base.predict
predict_proba = _base.predict_proba
