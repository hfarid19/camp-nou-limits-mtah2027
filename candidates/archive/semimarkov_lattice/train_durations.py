#!/usr/bin/env python3
"""Duration and transition tables for the semi-Markov decoder, from the DFL tracking flags.

    python -I candidates/semimarkov_lattice/train_durations.py --npz DIR --folds-out DIR

Reads the 7 parsed DFL/IDSSE games (<DIR>/<match>.npz as written by
learned_dfl_evidence/fetch_idsse.py: status 1 alive / 0 dead, poss 1 home / 2 away, half) and
labels each frame D where BallStatus is 0, else H/A from BallPossession (the bench's rule).
Per graded half it cuts the labels into spells and estimates

  * a duration log-pmf per class (possession spell, stoppage) over 1..DMAX frames: a Gaussian
    kernel density in log-seconds (bandwidth BW), normalised over integer frame counts;
  * transition probabilities: possession -> other team vs -> stoppage; stoppage -> restart by
    the team that was in possession before it ("same") vs the other team; first label of a half.

durations.npz (next to this file) uses all 7 games. For bench --dfl every game index gi gets
fold_dur_<gi>.npz built WITHOUT game gi (gi = position in sorted(os.listdir(npz dir))), and the
folds JSON maps durations.npz's sha256 prefix to that folder; the pipeline loads the table only
through dfl_folds.resolve().
"""

import argparse
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 25
DMAX = 400 * FPS        # longest segment the decoder considers (frames)
BW = 0.25               # KDE bandwidth in log-seconds at 1000 spells (scaled by n^-1/5)


def spells(path):
    z = np.load(path)
    st, po, half = z["status"], z["poss"], z["half"]
    c = np.full(len(st), 3, np.int8)
    c[(st == 1) & (po == 1)] = 0
    c[(st == 1) & (po == 2)] = 1
    c[st == 0] = 2
    out = []
    for h in (1, 2):
        idx = np.where((half == h) & (c != 3))[0]
        if not len(idx):
            continue
        x = c[idx[0]:idx[-1] + 1]
        ch = np.r_[0, np.where(np.diff(x) != 0)[0] + 1, len(x)]
        out.append([(int(x[a]), int(b - a)) for a, b in zip(ch[:-1], ch[1:])])
    return out


def kde_logpmf(durs_frames):
    d = np.arange(1, DMAX + 1) / FPS
    x = np.log(np.asarray(durs_frames, float) / FPS)
    bw = BW * (max(len(x), 1) / 1000.0) ** -0.2
    ld = np.log(d)
    dens = np.zeros(len(d))
    for chunk in np.array_split(x, max(1, len(x) // 500)):
        dens += np.exp(-0.5 * ((ld[:, None] - chunk[None, :]) / bw) ** 2).sum(1)
    dens /= d                         # density in log-space -> density in seconds
    dens = np.maximum(dens, dens.max() * 1e-12)
    pmf = dens / dens.sum()
    return np.log(pmf)


CTX = ["DD", "DX", "XD", "XX"]      # possession context: (entered from D / hand-over X, left to D / X)


def table(games):
    poss, dead = [], []
    ctx = {k: [] for k in CTX}
    exit_d = np.zeros((2, 2))         # [entry D / X, exit D / X] counts (uncensored spells)
    n_ho = n_pd = n_same = n_other = 0
    first = np.zeros(3)
    for g in games:
        for half in g:
            first[half[0][0]] += 1
            for i, (lab, d) in enumerate(half):
                (dead if lab == 2 else poss).append(d)
                if lab < 2 and 0 < i < len(half) - 1:
                    e = 0 if half[i - 1][0] == 2 else 1
                    x = 0 if half[i + 1][0] == 2 else 1
                    ctx[CTX[2 * e + x]].append(d)
                    exit_d[e, x] += 1
                if i + 1 < len(half):
                    nxt = half[i + 1][0]
                    if lab < 2:
                        if nxt == 2:
                            n_pd += 1
                        else:
                            n_ho += 1
                    elif 0 < i:
                        if nxt == half[i - 1][0]:
                            n_same += 1
                        else:
                            n_other += 1
    first = (first + 1) / (first.sum() + 3)
    return dict(logpmf=np.stack([kde_logpmf(poss), kde_logpmf(dead)]).astype(np.float64),
                p_handover=np.float64((n_ho + 1) / (n_ho + n_pd + 2)),
                p_restart_other=np.float64((n_other + 1) / (n_same + n_other + 2)),
                p_first=first,
                logpmf_ctx=np.stack([kde_logpmf(ctx[k]) for k in CTX]).astype(np.float64),
                p_exit_dead=(exit_d[:, 0] + 1) / (exit_d.sum(1) + 2), fps=np.int64(FPS), n_spells=np.array([len(poss), len(dead)]))


def save(path, t):
    np.savez(path, **t)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--folds-out", required=True)
    ap.add_argument("--folds-json", default=None, help="write/extend this folds JSON")
    a = ap.parse_args()
    matches = sorted(os.listdir(a.npz))
    games = [spells(os.path.join(a.npz, m)) for m in matches]
    os.makedirs(a.folds_out, exist_ok=True)
    sha = save(os.path.join(HERE, "durations.npz"), table(games))
    for gi in range(len(games)):
        save(os.path.join(a.folds_out, "fold_dur_{}.npz".format(gi)),
             table([g for k, g in enumerate(games) if k != gi]))
    print("durations.npz sha256", sha[:12], "folds", len(games), "->", a.folds_out)
    if a.folds_json:
        folds = {}
        if os.path.isfile(a.folds_json):
            folds = json.load(open(a.folds_json))
        folds = {k: v for k, v in folds.items() if v[1] != "dur"}
        folds[sha[:12]] = [os.path.abspath(a.folds_out), "dur"]
        with open(a.folds_json, "w") as handle:
            json.dump(folds, handle, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
