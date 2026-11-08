"""Collector (offline fitting only, never the submission): runs ONE member (env ENSV_MEMBER) through
members.py exactly as the ensemble does, returns its own codes / P(H,A,D), and also writes its codes
(and, for learned_dfl_evidence, the decoder side information) to ENSV_COLLECT/<tag>.npz so the
ensemble can be fitted offline. Under bench --dfl the tag is dfl_<match>_2027 (match = sorted DFL
list[BENCH_DFL_FOLD]); otherwise the n-th distinct input gets the n-th name in ENSV_TAGS.
"""

import os
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import members  # noqa: E402
WORK = os.environ.get("MTAH_WORK") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)).split(os.sep + "candidates")[0], "external", "work")

SLUG = os.environ["ENSV_MEMBER"]
OUT = os.environ["ENSV_COLLECT"]
DFL_DIR = WORK + "/idsse/npz"
MEMBER = members.Member(SLUG)
_SEEN = []
_RES = {}


def _tag(key):
    gi = os.environ.get("BENCH_DFL_FOLD")
    if gi is not None:
        matches = sorted(f[:-4] for f in os.listdir(DFL_DIR) if f.endswith(".npz"))
        return "dfl_{}_2027".format(matches[int(gi)])
    if key not in _SEEN:
        _SEEN.append(key)
    tags = os.environ.get("ENSV_TAGS", "").split(",")
    i = _SEEN.index(key)
    return tags[i] if i < len(tags) and tags[i] else "game{}".format(i)


def _run(args):
    key = members.input_key(args)
    if key not in _RES:
        _RES.clear()
        codes, proba, side = MEMBER.run(args)
        _RES[key] = (codes, proba)
        os.makedirs(OUT, exist_ok=True)
        extra = {k: v for k, v in (side or {}).items()}
        np.savez_compressed(os.path.join(OUT, _tag(key) + ".npz"), codes=codes,
                            proba=proba.astype(np.float32), **extra)
    return _RES[key]


def predict(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[0]


def predict_proba(players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps):
    return _run((players, slot_team, ball, sign_half1, blocks, tags, names, pitch, fps))[1]
