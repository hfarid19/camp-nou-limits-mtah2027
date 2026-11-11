"""install.py VARIANT d=<train_dir|lde> ha=<...> c=<...>: copy models into the candidate, update
model_meta.json (sha256 + per-model provenance) and write folds_<VARIANT>.json."""
import hashlib, json, os, shutil, sys
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")
CAND = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
LDE = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "learned_dfl_evidence"))
A = WORK + "/arch/skillcorner_domain"
VER = json.load(open(WORK + "/folds_verified.json"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
var = sys.argv[1]
spec = dict(a.split("=") for a in sys.argv[2:])
meta = json.load(open(os.path.join(CAND, "model_meta_lde.json")))
meta["variant"] = var
meta["source"] = {}
folds = {}
for w in ("d", "ha", "c"):
    src = spec.get(w, "lde")
    dst = os.path.join(CAND, "model_{}.txt".format(w))
    if src == "lde":
        shutil.copyfile(os.path.join(LDE, "model_{}.txt".format(w)), dst)
        meta["source"][w] = "learned_dfl_evidence (DFL only)"
        s = sha(dst)
        k = next(k for k in VER if s.startswith(k))
        folds[k] = VER[k]
    else:
        shutil.copyfile(os.path.join(src, "model_{}.txt".format(w)), dst)
        tm = json.load(open(os.path.join(src, "train_meta.json")))
        meta[w] = {k: v for k, v in tm[w].items()}
        meta["source"][w] = "train_sc.py DFL + SkillCorner ({})".format(os.path.basename(src))
        s = sha(dst)
        assert s == tm[w]["sha256"]
        folds[s[:12]] = [tm[w]["folds"], w]
    meta["sha256"][w] = s
json.dump(meta, open(os.path.join(CAND, "model_meta.json"), "w"), indent=1)
json.dump(folds, open(os.path.join(A, "folds_{}.json".format(var)), "w"), indent=1)
print(var, {w: meta["source"][w] for w in meta["source"]}, folds)
