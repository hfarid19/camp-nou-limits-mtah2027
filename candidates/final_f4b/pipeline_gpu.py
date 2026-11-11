"""EXPERIMENTS ONLY: final_f4b with the pointer, MS-TCN and TCN members' arithmetic on the GPU (candidates/gpu_accel.py).
Same members, weights and decoder as pipeline.py; outputs can differ from it in a handful of frames (float32).
Never the submission. GPU_ACCEL_BASE=pipeline_nopointer.py runs another pipeline of this folder the same way."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import gpu_accel  # noqa: E402

_spec = importlib.util.spec_from_file_location("final_f4b_pipeline_gpu_base",
                                               os.path.join(HERE, os.environ.get("GPU_ACCEL_BASE", "pipeline.py")))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
gpu_accel.install()


def predict(*a):
    gpu_accel.install()
    return _base.predict(*a)


def predict_proba(*a):
    gpu_accel.install()
    return _base.predict_proba(*a)
