"""nn_pointer, ball-centred variant r5_ptr_ballc (ballc.py: the 8 tracked players nearest the ball, coordinates relative
to the ball, ball kinematics; pointer over them + dead; train_ptr.py cfg ballc=1, FS 10, FF 14, Lp 0), 48 PFF + 7 DFL
games (ds_pff2). pipeline.py executed under a private module name with its weights pointed at ptr_ballc.npz (still
loaded through pff_folds.resolve, so the DFL / PFF benches use that file's fold models)."""

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nn_pointer_variant_ballc", os.path.join(HERE, "pipeline.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.WEIGHTS = os.path.join(HERE, "ptr_ballc.npz")

predict = _base.predict
predict_proba = _base.predict_proba
