#!/usr/bin/env python3
"""Write <dir>/pff_folds.json: sha256 prefix of <dir>/<anchor> (default fold_pff_0.npz; later the
shipped full model) -> <dir>, the mapping run_pff.py / pff_folds.resolve use to find fold_pff_<k>.npz.

    python candidates/nn_pff/folds_json.py DIR [ANCHOR ...]
"""
import hashlib
import json
import os
import sys

d = os.path.abspath(sys.argv[1])
anchors = sys.argv[2:] or ["fold_pff_0.npz"]
p = os.path.join(d, "pff_folds.json")
m = json.load(open(p)) if os.path.isfile(p) else {}
for a in anchors:
    f = a if os.path.isabs(a) else os.path.join(d, a)
    m[hashlib.sha256(open(f, "rb").read()).hexdigest()[:12]] = d
json.dump(m, open(p, "w"), indent=1)
print(p, m)
