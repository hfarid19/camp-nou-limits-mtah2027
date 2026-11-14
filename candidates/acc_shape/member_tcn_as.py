"""The same TCN (tcn_as.npz) without test-time averaging or re-normalisation: nn_pff_tcn/pipeline.py with the 223-column
features. Not a v11 member; it is the form the PFF benchmark numbers used for the TCN slot (as for v8 / v9)."""
import _common

_mod = _common.tcn_plain()
predict = _mod.predict
predict_proba = _mod.predict_proba
