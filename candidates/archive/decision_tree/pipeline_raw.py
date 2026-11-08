"""decision_tree without the control rule (CONTROL_S = 0): the tree + LDE's decoder only.

The bench variant for comparison with learned_dfl_evidence / ensemble_vote; pipeline.py
(CONTROL_S = 2.0) is the candidate.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("decision_tree_pipeline_base", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
sys.modules["decision_tree_pipeline_base"] = _base
_spec.loader.exec_module(_base)
_base.CONTROL_S = 0.0

predict = _base.predict
predict_proba = _base.predict_proba
