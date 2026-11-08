"""Nested leave-one-game-out selection of the vote's members / weights / few knobs on DFL only.

For every held-out DFL game g the whole selection below runs on the other 6 games' collected member
outputs; the chosen rule is then scored on g. The mean over the 7 held-out games is the honest
estimate. The same selection on all 7 games gives the shipped configuration.

    python fit.py hard     # (a) hard majority vote: member set x smoothing
    python fit.py soft     # (b) soft vote: pooling x weights x decoder variant x member set

Member order (tie-break order, top-k lists, greedy seeds) is always the ranking by mean spellF1 on
the TRAINING games of the fold.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fitlib as F   # noqa: E402

OUT = os.path.join(F.SCRATCH, "arch", "ensemble_vote", "fit")
os.makedirs(OUT, exist_ok=True)

SMOOTH = {"none": None, "m05": dict(min_poss_s=0.5, min_dead_s=0.5),
          "m10": dict(min_poss_s=1.0, min_dead_s=1.0)}
DEC = {"c0": dict(contrast_beta=0.0, w_gap=0.0),       # members' evidence as is
       "c1": dict(contrast_beta=1.25, w_gap=0.0),      # + LDE's local-contrast team logit
       "c2": dict(contrast_beta=1.25, w_gap=3.0)}      # + LDE's long-gap dead term (LDE default)


def member_order(sp, train):
    return sorted(sp.slugs, key=lambda s: -np.mean([sp.member_scores[s][t]["spell"] for t in train]))


def weights_for(sp, slugs, train, wmode):
    if wmode == "eq":
        return None
    return [float(np.mean([sp.member_scores[s][t]["spell"] for t in train])) for s in slugs]


class Evaluator(object):
    def __init__(self, sp, kind):
        self.sp, self.kind = sp, kind
        self.cache = F.Cache(os.path.join(OUT, "cache_{}_{}.json".format(kind, sp.split)))

    def game(self, t, slugs, opt, weights=None):
        key = json.dumps([t, list(slugs), opt, None if weights is None else
                          [round(w, 6) for w in weights]])

        def run():
            if self.kind == "hard":
                pred = self.sp.hard(t, slugs, SMOOTH[opt["smooth"]], weights)
            else:
                pred = self.sp.soft(t, slugs, opt["pool"], weights, DEC[opt["dec"]])
            return F.score(self.sp.gt[t], pred)
        return self.cache.get(key, run)

    def mean(self, games, slugs, opt, train):
        w = weights_for(self.sp, slugs, train, opt.get("w", "eq"))
        return float(np.mean([self.game(t, slugs, opt, w)["spell"] for t in games]))


def stepwise(ev, train, order, opt, start, max_steps=12):
    """Add-or-remove-one stepwise search from `start`, keeping the training order (tie-break)."""
    cur = list(start)
    best = ev.mean(train, cur, opt, train)
    for _ in range(max_steps):
        cands = []
        for s in order:
            nxt = [m for m in order if (m in cur) != (m == s)]
            if len(nxt) == 0:
                continue
            cands.append((ev.mean(train, nxt, opt, train), nxt))
        sc, nxt = max(cands, key=lambda c: c[0])
        if sc <= best + 1e-9:
            break
        best, cur = sc, nxt
    return best, cur


def select(ev, train, smooth_opts=None):
    order = member_order(ev.sp, train)
    tried = []
    if ev.kind == "hard":
        for sm in smooth_opts:
            opt = dict(smooth=sm)
            for k in (3, 5, 7, 9, len(order)):
                tried.append((ev.mean(train, order[:k], opt, train), order[:k], opt, "top%d" % k))
            sc, cur = stepwise(ev, train, order, opt, order[:3])
            tried.append((sc, cur, opt, "stepwise"))
    else:
        combos = [dict(pool=p, w=w, dec=d) for p in ("arith", "geo") for w in ("eq", "score")
                  for d in ("c0", "c1", "c2")]
        stage1 = []
        for opt in combos:
            for k in (3, len(order)):
                stage1.append((ev.mean(train, order[:k], opt, train), order[:k], opt, "top%d" % k))
        tried += stage1
        best_opt = max(stage1, key=lambda c: c[0])[2]
        for k in (1, 2, 4, 5, 7):
            tried.append((ev.mean(train, order[:k], best_opt, train), order[:k], best_opt,
                          "top%d" % k))
        start = max([c for c in tried if c[2] == best_opt], key=lambda c: c[0])[1]
        sc, cur = stepwise(ev, train, order, best_opt, start)
        tried.append((sc, cur, best_opt, "stepwise"))
    best = max(tried, key=lambda c: c[0])
    return best, order


def nested(ev, smooth_opts=None, label=""):
    sp = ev.sp
    rows = []
    for g in sp.tags:
        train = [t for t in sp.tags if t != g]
        (tr, slugs, opt, how), order = select(ev, train, smooth_opts)
        w = weights_for(sp, slugs, train, opt.get("w", "eq"))
        r = ev.game(g, slugs, opt, w)
        best_single = order[0]
        rows.append(dict(game=g, held=r["spell"], macro=r["macro"], poss="%d/%d" % (r["n_pred"], r["n_gt"]),
                         train=tr, slugs=slugs, opt=opt, how=how, best_single=best_single,
                         best_single_held=sp.member_scores[best_single][g]["spell"]))
        print("  {} held {:.4f} (best single on train {} -> {:.4f}) train {:.4f} {} {} {}".format(
            g[4:-5], r["spell"], best_single, rows[-1]["best_single_held"], tr, how, opt,
            ",".join(slugs)), flush=True)
    ev.cache.save()
    held = float(np.mean([r["held"] for r in rows]))
    single = float(np.mean([r["best_single_held"] for r in rows]))
    (tr, slugs, opt, how), order = select(ev, list(sp.tags), smooth_opts)
    ev.cache.save()
    final = dict(slugs=slugs, opt=opt, how=how, in_sample=tr, order=order)
    print("NESTED {} {}: {:.4f} (nested best-single {:.4f}); final on all 7: {:.4f} {} {} {}".format(
        ev.kind, label, held, single, tr, how, opt, ",".join(slugs)), flush=True)
    return dict(kind=ev.kind, label=label, nested=held, nested_best_single=single, rows=rows,
                final=final)


def main():
    kind = sys.argv[1]
    sp = F.Split("dfl", F.members.ALL)
    print("members:", ", ".join("{} {:.4f}".format(s, np.mean([v["spell"] for v in sp.member_scores[s].values()]))
                                for s in sp.slugs))
    ev = Evaluator(sp, kind)
    res = []
    if kind == "hard":
        res.append(nested(ev, ["none"], "no smoothing"))
        res.append(nested(ev, ["m05", "m10"], "smoothing selected (m05/m10)"))
        res.append(nested(ev, ["none", "m05", "m10"], "smoothing incl. none"))
    else:
        res.append(nested(ev, None, "soft"))
    json.dump(res, open(os.path.join(OUT, "nested_{}.json".format(kind)), "w"), indent=1)


if __name__ == "__main__":
    main()
