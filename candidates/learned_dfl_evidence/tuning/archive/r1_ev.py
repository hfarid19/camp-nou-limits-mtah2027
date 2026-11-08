import os, sys, pickle, importlib.util, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm
HERE = os.environ.get("LDE_TUNE_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "tune"))
os.makedirs(HERE, exist_ok=True)
def load_lde(path=os.path.join(ROOT, "candidates/learned_dfl_evidence/pipeline.py")):
    spec = importlib.util.spec_from_file_location("lde", path)
    L = importlib.util.module_from_spec(spec); sys.path.insert(0, os.path.dirname(path)); spec.loader.exec_module(L)
    return L
L = load_lde()
_R = {}
def rows(mode):
    if mode not in _R: _R[mode] = pickle.load(open(os.path.join(HERE, "rows_%s.pkl" % mode if "/" not in mode else "rows_%s.pkl" % mode.replace("/", "")), "rb"))
    return _R[mode]
def score(mode, cfg=None, fn=None):
    out = []
    c = dict(L.CFG, **(cfg or {}))
    for r in rows(mode):
        aux = dict(r["aux"])
        aux["long_gap"] = L.long_gap_gated(r["players"], r["ball"], 25, c["gap_s"], c["gap_speed_ms"])
        if fn is None:
            pred = L.decode(r["pdd"], r["ph"], aux, r["blocks"], len(r["ball"]), 25, cfg=cfg)
        else:
            pred = fn(r, aux, c)
        e = pm.evaluate(r["gt"], np.asarray(pred, np.int8), bootstrap_n=0)
        out.append((e["spell_f1_mean"], e["macro_f1"], e["n_pred_spells"]))
    a = np.array(out)
    return round(a[:, 0].mean(), 4), round(a[:, 1].mean(), 4), int(a[:, 2].sum() / 4)
if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    modes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["clean", "osd", "offset"]
    print(cfg, {m: score(m, cfg) for m in modes}, flush=True)
