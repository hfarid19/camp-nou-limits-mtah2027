"""Bench wrapper for SCREENING a configuration with the torch inference backend (asf_torch.py; round 4).

= pipeline_cfg.py ($NN_ASF_MODEL, weights through pff_folds.resolve, so run_pff.py / bench --dfl2 use the
fold models) with asf_np.Model replaced by asf_torch.Model (float32, chunked attention, ~7x faster on CPU
for the full-attention arch; parity with numpy ~4e-7 on probabilities). Not a submission path: shipping
needs numpy (asf_np) inference.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asf_torch  # noqa: E402

_spec = importlib.util.spec_from_file_location(os.path.basename(HERE) + "_torch_variant", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_attr = "MODEL_PATH" if hasattr(_base, "MODEL_PATH") else "WEIGHTS"
setattr(_base, _attr, os.path.abspath(os.environ["NN_ASF_MODEL"]))
_base.asf_np = asf_torch          # tcn_outputs looks up asf_np.Model at call time

predict = _base.predict
predict_proba = _base.predict_proba
boundary_proba = getattr(_base, "boundary_proba", None)
