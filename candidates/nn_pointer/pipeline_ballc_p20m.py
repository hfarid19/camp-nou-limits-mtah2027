"""nn_pointer, ball-centred variant widened to 19,258,736 parameters: r5_ptr_ballc_p20M (round-5 pointer size ladder;
train_ptr.py cfg ballc=1, FS 10, FF 14, Lp 0, D 320, heads 20, C 640, Dp 160 = r5_ptr_ballc's widths x 5, everything
else unchanged), 48 PFF + 7 DFL games (ds_pff2). Every width is read from the model's cfg (ptr_np.forward), so the
code is pipeline_ballc.py's: pipeline.py executed under a private module name with its weights pointed at
ptr_ballc_p20m.npz (still loaded through pff_folds.resolve, so the DFL / PFF benches use that file's fold models).
Numpy float64 on the CPU: about 24 minutes for a 200k-frame match (pipeline_ballc.py: about 75 s)."""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nn_pointer_variant_ballc_p20m", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.WEIGHTS = os.path.join(HERE, "ptr_ballc_p20m.npz")

predict = _base.predict
predict_proba = _base.predict_proba
