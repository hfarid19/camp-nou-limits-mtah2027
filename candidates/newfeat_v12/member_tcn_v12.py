"""TCN member of v12 without test-time averaging (the PFF bench's TCN path, as acc_shape/member_tcn_as.py)."""
import _common

_mod = _common.tcn_plain()
predict = _mod.predict
predict_proba = _mod.predict_proba
