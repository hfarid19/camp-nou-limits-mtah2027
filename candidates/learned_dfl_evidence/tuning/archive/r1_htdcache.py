import os, sys, pickle, importlib.util
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
HERE = os.environ.get("LDE_TUNE_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "tune"))
os.makedirs(HERE, exist_ok=True)
spec = importlib.util.spec_from_file_location("htd", os.path.join(ROOT, "candidates/hier_touch_decoder/pipeline.py"))
htd = importlib.util.module_from_spec(spec); sys.path.insert(0, os.path.join(ROOT, "candidates/hier_touch_decoder")); spec.loader.exec_module(htd)
mode = sys.argv[1]
rows = pickle.load(open(os.path.join(HERE, "rows_%s.pkl" % mode), "rb"))
C = dict(htd.CFG, w_lde_dead=0.0)
out = []
for r in rows:
    Fe = htd.features(r["players"], r["st"], r["ball"], r["sign"], r["blocks"], [], ("Home", "Away"), (105.0, 68.0) if False else __import__("tune_on_sample_data").PITCH, 25, C)
    dl = htd.dead_logit(Fe, C)
    m = np.zeros(len(r["ball"]), bool)
    for _, f, l in r["blocks"]:
        m[f:l + 1] = htd.stage_a(Fe, f, l, C, dl)
    out.append(dict(dl=dl, mask=m))
pickle.dump(out, open(os.path.join(HERE, "htd_%s.pkl" % mode), "wb"))
print("ok", mode)
