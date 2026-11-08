#!/usr/bin/env python3
"""Assemble a model set: copy the d / ha / c model files into OUT and write OUT/model_meta.json
(file, feature names read from the booster, sha256, lightgbm version) for pipeline.py.

    python -I assemble.py --d FILE --ha FILE --c FILE --out DIR
"""
import argparse, hashlib, json, os, shutil
import lightgbm as lgb

ap = argparse.ArgumentParser()
for w in ("d", "ha", "c"):
    ap.add_argument("--" + w, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
meta = dict(lgb_version=lgb.__version__)
for w in ("d", "ha", "c"):
    src = getattr(a, w)
    dst = os.path.join(a.out, "model_{}.txt".format(w))
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
    b = lgb.Booster(model_file=dst)
    meta[w] = dict(file=os.path.basename(dst), names=b.feature_name(), rounds=b.num_trees(),
                   sha256=hashlib.sha256(open(dst, "rb").read()).hexdigest())
    print(w, src, meta[w]["sha256"][:12], len(meta[w]["names"]), "features", b.num_trees(), "trees")
with open(os.path.join(a.out, "model_meta.json"), "w") as h:
    json.dump(meta, h, indent=1)
