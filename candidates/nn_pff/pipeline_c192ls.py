"""nn_pff, "C192ls" variant: the m5c recipe at width C 192 / Cr 96 with label smoothing 0.05 (round-4 config m5c_C192ls), DFL + 48 PFF games (inputs as
ds_v2s: degrade_v2 only), weights mstcn_c192ls.npz; see candidates/nn_pff/README.md.

pipeline.py executed under a private module name with its weights pointed at mstcn_c192ls.npz (still
loaded through pff_folds.resolve, so the DFL / PFF benches use that file's fold models).
"""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nn_pff_variant_c192ls", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
setattr(_base, "MODEL_PATH" if hasattr(_base, "MODEL_PATH") else "WEIGHTS", os.path.join(HERE, "mstcn_c192ls.npz"))

predict = _base.predict
predict_proba = _base.predict_proba
