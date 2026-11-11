"""EXPERIMENTS ONLY (pointer size ladder scoring): pipeline_cfg.py (weights $NN_PFF_MODEL, fold models through
pff_folds.resolve) with ptr_np.forward run on the GPU by candidates/gpu_accel.py (float32 torch port, as
final_f4b/pipeline_gpu.py does). Needs PYTORCH_ENABLE_MPS_FALLBACK=1 on the Mac. Outputs can differ from the numpy
path in the last float32 digits: never the submission; codes checked equal to the CPU path on one game per model."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import gpu_accel  # noqa: E402

_spec = importlib.util.spec_from_file_location("nn_pointer_cfg_gpu_base", os.path.join(HERE, "pipeline_cfg.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
gpu_accel.install()


def _start_worker():
    """Start gpu_accel's spawn worker with candidates/nn_pff (its queue.py shadows the stdlib queue that torch
    imports) left out of the sys.path the child inherits; otherwise identical to gpu_accel._call's start."""
    if gpu_accel._CONN is not None:
        return
    import multiprocessing as mp
    saved = list(sys.path)
    nn_pff = os.path.join(os.path.dirname(HERE), "nn_pff")
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != nn_pff]
    try:
        ctx = mp.get_context("spawn")
        gpu_accel._CONN, child = ctx.Pipe()
        ctx.Process(target=gpu_accel._serve, args=(child,), daemon=True).start()
    finally:
        sys.path[:] = saved


def predict(*a):
    _start_worker()
    gpu_accel.install()
    return _base.predict(*a)


def predict_proba(*a):
    _start_worker()
    gpu_accel.install()
    return _base.predict_proba(*a)
