"""dtree_simple with the challenge's control rule on its output: control.fold at CONTROL_S = 2.0 s
(the value ensemble_ctrl picked on DFL leave-one-game-out; not re-tuned here)."""

import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("dtree_simple_base", os.path.join(_HERE, "pipeline.py"))
_BASE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_BASE)
_BASE.CONTROL_S = 2.0

predict = _BASE.predict
predict_proba = _BASE.predict_proba
