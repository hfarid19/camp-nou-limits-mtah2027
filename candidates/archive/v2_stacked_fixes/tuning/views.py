"""Robustness views for v2 tuning (dev bench rows only: Sample Games 1, 2 x seeds 2027, 2028).

Views (all fixed seeds, deterministic):
  clean    the bench as is
  offset   AR(1) ball offset, sd 1.3 m per axis, tau 25 frames, seed 7
  osd      smooth (9-frame NaN-aware mean) + the same offset + chunked player dropout (seed 8);
           copied from the round-1 judges' stress wrapper
  stress2  out-of-distribution: AR(1) sd 2 m, tau 75 frames (seed 11) plus extra 0.5-2 s ball
           dropouts; copied from the round-2 measured-performance judge (cfgrun3.py)

Usage: python tuning/views.py '{"cfg_key": value, ...}' [label]
Prints spellF1 / macroF1 per view and the mean of clean, osd and stress2.
Never touches evaluation/gt, the holdout game or the competition match.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "candidates"))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("V2_QUIET", "1")
import bench  # noqa: E402
import pipeline  # noqa: E402
import possession_pipeline as pp  # noqa: E402

pm, tsd = bench.pm, bench.tsd

_MEMO = {}
_lde_raw = pipeline._lde_logits


def _lde_memo(players, slot_team, ball, *a):
    """Sweep speed-up only: the frozen-model outputs depend on the inputs alone."""
    import hashlib
    key = (hashlib.md5(players.tobytes()).hexdigest(), hashlib.md5(ball.tobytes()).hexdigest(), a[-1])
    if key not in _MEMO:
        _MEMO[key] = _lde_raw(players, slot_team, ball, *a)
    return _MEMO[key]


pipeline._lde_logits = _lde_memo


def _offset(ball, rng, sd=1.3, tau=25):
    n = len(ball)
    a = np.exp(-1.0 / tau)
    e = rng.normal(0, sd * np.sqrt(1 - a * a), (n, 2))
    off = np.zeros((n, 2))
    off[0] = rng.normal(0, sd, 2)
    for t in range(1, n):
        off[t] = a * off[t - 1] + e[t]
    return (ball + off).astype(np.float32)


def _smooth(ball, hw=4):
    seen = ~np.isnan(ball[:, 0])
    out = np.stack([pp.moving_mean(ball[:, i].astype(np.float64), hw) for i in (0, 1)], 1)
    out[~seen] = np.nan
    return out.astype(np.float32)


def _drop(players, rng, rate=0.06, mean_len=50):
    players = players.copy()
    n, S, _ = players.shape
    starts = rng.random((n, S)) < rate / mean_len
    for s, j in zip(*np.where(starts)):
        players[s:s + int(rng.exponential(mean_len)) + 1, j] = np.nan
    return players


def view(name, players, ball):
    if name == "clean":
        return players, ball
    if name == "offset":
        return players, _offset(ball.copy(), np.random.default_rng(7))
    if name == "osd":
        b = _offset(_smooth(ball), np.random.default_rng(7))
        return _drop(players, np.random.default_rng(8)), b
    if name == "stress2":
        rng = np.random.default_rng(11)
        b = _offset(ball.copy(), rng, sd=2.0, tau=75)
        for t in np.flatnonzero(rng.random(len(b)) < 0.2 / 30):
            b[t:t + int(rng.integers(12, 50))] = np.nan
        return players, b
    raise ValueError(name)


VIEWS = ("clean", "offset", "osd", "stress2")


def run(cfg, views=VIEWS, predict=None):
    predict = predict or pipeline.predict
    res = {}
    for v in views:
        sf, mf, npred = [], [], []
        for game in bench.DEV:
            for seed in bench.SEEDS:
                slots, st, ball, gt, blocks, sign = bench.prepared(game, seed)
                pl, b = view(v, slots, ball)
                p = np.asarray(predict(pl, st, b, sign, blocks, [], ("Home", "Away"), tsd.PITCH,
                                       tsd.FPS, cfg=cfg), np.int8)
                r = pm.evaluate(gt, p, bootstrap_n=0)
                sf.append(r["spell_f1_mean"])
                mf.append(r["macro_f1"])
                npred.append(r["n_pred_spells"])
        res[v] = (float(np.mean(sf)), float(np.mean(mf)), [round(x, 4) for x in sf], npred)
    return res


def report(cfg, label=None):
    label = label or json.dumps(cfg)
    res = run(cfg)
    parts = ["{} {:.4f}/{:.4f}".format(v, res[v][0], res[v][1]) for v in res]
    m3 = np.mean([res[v][0] for v in ("clean", "osd", "stress2") if v in res])
    print("{:34s} {} | mean(clean,osd,s2) {:.4f} | clean rows {} poss {}".format(
        label, " | ".join(parts), m3, res["clean"][2], res["clean"][3]), flush=True)
    return res


if __name__ == "__main__":
    # one json cfg per argument; each line is one configuration
    for arg in sys.argv[1:] or ["{}"]:
        report(json.loads(arg))
