"""Bench wrapper (experiments): nn_seg/pipeline.py under a private module name with its weights pointed at
$NN_PFF_MODEL (a config's full.npz); fold models still load through pff_folds.resolve."""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nn_seg_cfg_variant", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.MODEL_PATH = os.path.abspath(os.environ["NN_PFF_MODEL"])

predict = _base.predict
predict_proba = _base.predict_proba
