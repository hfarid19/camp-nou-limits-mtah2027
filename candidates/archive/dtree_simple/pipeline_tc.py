"""dtree_simple with the home/away tree trained on control-folded targets (control.fold 2.0 s on the
full-resolution DFL codes, train_tree.py --target ctrl). No output fold."""

import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("dtree_simple_tc_base", os.path.join(_HERE, "pipeline.py"))
_BASE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_BASE)
_BASE.TARGET = "ctrl"

predict = _BASE.predict
predict_proba = _BASE.predict_proba
