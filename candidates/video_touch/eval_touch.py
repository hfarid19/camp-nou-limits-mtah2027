"""Evaluate per-frame touch probabilities on held-out halves.
usage: eval_touch.py RUN_DIR [halves...]  (default: all prob_*.npy in RUN_DIR)
Reports: AP@+-2/+-5, best-F1 operating points, P/R at fixed thresholds, a chance baseline
(prediction circularly shifted by a random 10-60 s; same peaks, timing destroyed), and the
signed offset distribution of matched peaks (alignment check)."""
import sys, os, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_common import events, TOUCH
from vt_eval import nms_peaks, match, average_precision

run = sys.argv[1]
files = sorted(glob.glob(f'{run}/prob_*.npy'))
R = 6
rows = []
allp, allg = [], []
off_all = []
for f in files:
    m, h = os.path.basename(f)[5:-4].split('_')
    p = np.load(f)
    E = [e for e in events(m, '1' if h == '1st' else '2') if e['label'] in TOUCH]
    gt = np.array(sorted(e['frame'] - 1 for e in E))
    allp.append((p, gt))
    pk = nms_peaks(p, 0.5, R)
    _, pairs = match(pk, gt, 5)
    off_all += [int(pk[a] - gt[b]) for a, b in pairs]

def summarize(tol, thr_list=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    out = {}
    for thr in thr_list:
        tp = npk = ng = 0
        for p, gt in allp:
            pk = nms_peaks(p, thr, R); t, _ = match(pk, gt, tol)
            tp += t; npk += len(pk); ng += len(gt)
        out[thr] = (tp / max(npk, 1), tp / ng, npk)
    return out

for tol in (2, 5):
    aps = [average_precision(p, gt, tol, R) for p, gt in allp]
    rng = np.random.default_rng(0)
    chance = [average_precision(np.roll(p, int(rng.integers(250, 1500))), gt, tol, R) for p, gt in allp]
    print(f'tol +-{tol}: AP {np.mean(aps):.3f} (per half {np.round(aps,3).tolist()}), chance AP {np.mean(chance):.3f}')
    for thr, (P, Rr, n) in summarize(tol).items():
        print(f'   thr {thr:.1f}: P {P:.3f} R {Rr:.3f} F1 {2*P*Rr/max(P+Rr,1e-9):.3f} peaks {n}')
    # chance at the 0.5 threshold
    tp = npk = 0
    for p, gt in allp:
        pk = nms_peaks(np.roll(p, 777), 0.5, R); t, _ = match(pk, gt, tol); tp += t; npk += len(pk)
    print(f'   chance P at thr 0.5 (shifted): {tp/max(npk,1):.3f}')
o = np.array(off_all)
print('matched offsets (pred - gt) at thr .5, tol 5: n', len(o), 'mean %.2f median %d' % (o.mean(), np.median(o)),
      'hist', {k: int((o == k).sum()) for k in range(-5, 6)})
